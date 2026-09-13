/* ==========================================================================
   IBVAP — Border Surveillance Command Dashboard
   --------------------------------------------------------------------------
   Everything rendered here comes from the backend. There are no simulated
   detections, no fabricated statistics and no timers pretending to be events.

   Performance notes:
     * Alerts arrive over a single WebSocket; the feed is capped so a long
       shift cannot grow the DOM without bound.
     * Stats update by writing to existing nodes rather than re-rendering.
     * Camera tiles are built once and only their metric text is refreshed —
       rebuilding a tile would restart its MJPEG stream.
   ========================================================================== */
'use strict';

const IBVAP = (() => {

  const MAX_FEED = 60;                  // alert cards retained in the live feed
  const PAGE_SIZE = 50;

  /*
   * Live-stream connection budget.
   *
   * A browser opens at most 6 concurrent HTTP/1.1 connections per origin, and
   * an MJPEG tile holds one of them open forever — the response never ends,
   * which is the whole point of the format. With six tiles streaming, all six
   * slots are permanently occupied and every other request the dashboard makes
   * queues behind them and never runs: the 1 Hz stats poll, the event log, the
   * snapshot that seeds the fence canvas, even the DELETE behind the Remove
   * button. Measured: with 5 streams open /health answered in 4 ms; with 6 it
   * never answered at all, and closing one stream recovered it instantly.
   *
   * The symptom is nasty because it does not look like a network problem — the
   * already-open video keeps moving and WebSocket alerts keep arriving (a
   * separate pool), so the page looks alive while every control is dead.
   *
   * So live streams are capped below the limit and the remaining tiles refresh
   * from single-frame snapshots, which return their connection between frames.
   * Slower video on the overflow tiles, and a dashboard that keeps working.
   */
  const STREAM_BUDGET = 4;              // leaves 2 of the browser's 6 for the API
  const SNAPSHOT_INTERVAL_MS = 1000;    // refresh rate for tiles beyond the budget
  //: Digital-zoom bounds for a paused tile. Past about 8x on a 640x384 frame
  //: there are no more pixels to show, only bigger ones.
  const ZOOM_MIN = 1;
  const ZOOM_MAX = 8;

  const state = {
    cameras: [],
    tiles: new Map(),                   // camera_id -> DOM refs
    alerts: [],
    view: 'operations',
    ws: null,
    wsRetry: 0,
    logOffset: 0,
    logTotal: 0,
    alertTypes: new Set(),
    upload: { file: null, sessionId: null, poll: null },
    fence: { cameraId: null, cameraName: '', mode: 'line', points: [], rules: [], image: null },
    camera: { kind: 'live', file: null },
    frame: { width: 640, height: 384 },
    deviceLabel: '—',
    // Alerting. Both default OFF: a control room where the speakers start on
    // their own is a liability, and browsers block audio before a user gesture
    // anyway. The choice is remembered per browser.
    alerting: { sound: false, notify: false, wantedSound: false,
                lastNotifiedId: 0, lastZoneSoundAt: 0,
                // Filled from /api/system/info so retuning NOTIFY_MIN_SEVERITY
                // on the server is enough; the front end has no second copy.
                redSeverities: ['HIGH', 'CRITICAL'],
                severityOrder: { INFO: 0, LOW: 1, MEDIUM: 2, HIGH: 3, CRITICAL: 4 },
                minSeverity: 'HIGH' },
    zones: { occupied: 0, breached: 0, cameras: new Map() },
  };

  /* ----------------------------------------------------------- utilities */

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  /* --------------------------------------------------------- alert signals */

  /**
   * Audible alerting, synthesised rather than loaded.
   *
   * The tone is generated with the Web Audio API instead of shipping an .mp3
   * for three reasons that matter at a Border Out Post: there is no file to
   * fetch, so it works on a LAN with no internet and cannot be delayed by a
   * slow first load; it costs nothing in the page weight; and the pitch and
   * length can carry meaning — a short double beep for an ordinary event, a
   * lower urgent triple for a CRITICAL one, and a slow repeating pulse while a
   * zone stays occupied.
   *
   * The AudioContext is created lazily on the operator's click, because every
   * browser now refuses to start audio without a user gesture. Creating it at
   * load would produce a context stuck in "suspended" and silence with no
   * error — the classic "why is the alarm not working" bug.
   */
  const Sound = (() => {
    let ctx = null;

    function context() {
      if (!ctx) {
        const Ctor = window.AudioContext || window.webkitAudioContext;
        if (!Ctor) return null;
        ctx = new Ctor();
      }
      if (ctx.state === 'suspended') ctx.resume().catch(() => {});
      return ctx;
    }

    /** One beep: a sine partial plus a square partial so it cuts through room noise. */
    function beep(startAt, freq, seconds, gainPeak) {
      const audio = context();
      if (!audio) return;
      const t0 = audio.currentTime + startAt;
      const gain = audio.createGain();
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(gainPeak, t0 + 0.012);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + seconds);
      gain.connect(audio.destination);

      [['sine', freq, 1], ['square', freq * 2, 0.25]].forEach(([type, f, mix]) => {
        const osc = audio.createOscillator();
        osc.type = type;
        osc.frequency.setValueAtTime(f, t0);
        const sub = audio.createGain();
        sub.gain.value = mix;
        osc.connect(sub); sub.connect(gain);
        osc.start(t0);
        osc.stop(t0 + seconds + 0.02);
      });
    }

    /**
     * A rising-falling siren, for a CRITICAL alert only.
     *
     * A sweep rather than discrete beeps because a sweep is what the ear
     * reads as an alarm — it is the shape every emergency signal uses, and it
     * stays recognisable through room noise and a cheap speaker in a way a
     * short tone does not.
     */
    function siren(sweeps, gainPeak) {
      const audio = context();
      if (!audio) return;
      const t0 = audio.currentTime;
      const sweep = 0.55;
      const gain = audio.createGain();
      gain.gain.setValueAtTime(0.0001, t0);
      gain.gain.exponentialRampToValueAtTime(gainPeak, t0 + 0.05);
      gain.gain.setValueAtTime(gainPeak, t0 + sweeps * sweep - 0.08);
      gain.gain.exponentialRampToValueAtTime(0.0001, t0 + sweeps * sweep);
      gain.connect(audio.destination);

      const osc = audio.createOscillator();
      osc.type = 'sawtooth';                 // harmonically rich: it carries
      osc.frequency.setValueAtTime(520, t0);
      for (let n = 0; n < sweeps; n += 1) {
        osc.frequency.linearRampToValueAtTime(980, t0 + n * sweep + sweep / 2);
        osc.frequency.linearRampToValueAtTime(520, t0 + (n + 1) * sweep);
      }
      osc.connect(gain);
      osc.start(t0);
      osc.stop(t0 + sweeps * sweep + 0.05);
    }

    return {
      /** Verify the browser will actually make a noise, and say so. */
      prime() {
        const audio = context();
        if (!audio) return false;
        beep(0, 880, 0.07, 0.18);
        return true;
      },

      /**
       * Sound an event, graded by severity.
       *
       * Only a red alert makes a sound at all. Everything below it — a passing
       * car, a plate read, a face seen — is logged and shown and stays silent,
       * because a console that chirps at routine traffic is a console whose
       * operator stops hearing it.
       */
      event(severity) {
        if (!state.alerting.sound) return;
        const level = String(severity || '').toUpperCase();
        if (!state.alerting.redSeverities.includes(level)) return;

        if (level === 'CRITICAL') {
          siren(3, 0.42);                    // loudest, longest, unmistakable
        } else {
          beep(0.00, 780, 0.13, 0.30);       // HIGH: a firm two-tone alarm
          beep(0.18, 620, 0.18, 0.30);
        }
      },
      /** The continuous one: a slow pulse held while a zone stays occupied. */
      zonePulse(breached) {
        if (!state.alerting.sound) return;
        const now = Date.now();
        const gap = breached ? 1600 : 3200;   // urgent zones repeat faster
        if (now - state.alerting.lastZoneSoundAt < gap) return;
        state.alerting.lastZoneSoundAt = now;
        beep(0.00, breached ? 520 : 660, 0.14, breached ? 0.30 : 0.18);
        if (breached) beep(0.19, 415, 0.20, 0.30);
      },
    };
  })();

  /**
   * Desktop notifications, so an operator watching another window still knows.
   *
   * Permission is requested only when the operator turns the feature on — a
   * page that asks on load is the pattern browsers now auto-deny, which would
   * leave the feature permanently unavailable with no way back except digging
   * through site settings. Notifications are deduplicated by alert id and
   * tagged per camera, so a burst replaces itself in the tray instead of
   * stacking twenty cards.
   */
  const Notifier = {
    supported() { return 'Notification' in window; },

    async enable() {
      if (!this.supported()) return 'unsupported';
      if (Notification.permission === 'granted') return 'granted';
      if (Notification.permission === 'denied') return 'denied';
      try { return await Notification.requestPermission(); }
      catch { return 'denied'; }
    },

    /**
     * Raise a desktop notification — for a red alert, and nothing else.
     *
     * This is the difference between a system an operator trusts and one they
     * mute. On a road-facing camera the routine types dominate by an order of
     * magnitude (90 `vehicle_detected` rows in a 45-second run), and a tray
     * card for each of them buries the intrusion that arrives between them.
     * Severity is graded server-side and the threshold is served with it, so
     * this check and the log always agree on what counts.
     */
    show(alert) {
      if (!state.alerting.notify || !this.supported()) return;
      if (Notification.permission !== 'granted') return;
      const level = String(alert.severity || '').toUpperCase();
      if (!state.alerting.redSeverities.includes(level)) return;
      if (alert.id && alert.id <= state.alerting.lastNotifiedId) return;
      if (alert.id) state.alerting.lastNotifiedId = alert.id;
      try {
        const note = new Notification(`${alert.icon || '⚠️'} ${alert.title || alert.alert_type}`, {
          body: `${alert.camera_name || 'Camera'} · ${alert.severity || ''}\n${alert.description || ''}`.trim(),
          tag: `ibvap-cam-${alert.camera_id}`,   // one live card per camera
          renotify: true,
          requireInteraction: alert.severity === 'CRITICAL',
          silent: true,                          // our own tone is the sound
        });
        note.onclick = () => {
          window.focus();
          showView('events');
          note.close();
        };
        // Ordinary alerts clear themselves; a CRITICAL one is left for the
        // operator to dismiss, which is what requireInteraction above asks for.
        if (alert.severity !== 'CRITICAL') setTimeout(() => note.close(), 12000);
      } catch { /* a notification must never break the dashboard */ }
    },
  };

  function toggleSound() {
    state.alerting.sound = !state.alerting.sound;
    if (state.alerting.sound && !Sound.prime()) {
      state.alerting.sound = false;
      toast('This browser will not play audio.', 'err');
    } else {
      toast(state.alerting.sound ? 'Alert sound armed' : 'Alert sound muted', 'ok');
    }
    persistAlerting();
    renderAlertToggles();
  }

  async function toggleNotifications() {
    if (state.alerting.notify) {
      state.alerting.notify = false;
      toast('Desktop notifications off', 'ok');
    } else {
      const result = await Notifier.enable();
      if (result === 'granted') {
        state.alerting.notify = true;
        toast('Desktop notifications armed', 'ok');
      } else if (result === 'denied') {
        toast('Notifications are blocked for this site — allow them in the '
              + 'browser\u2019s site settings, then try again.', 'err', 8000);
      } else {
        toast('This browser does not support notifications.', 'err');
      }
    }
    persistAlerting();
    renderAlertToggles();
  }

  function persistAlerting() {
    try {
      localStorage.setItem('ibvap.alerting', JSON.stringify({
        sound: state.alerting.sound, notify: state.alerting.notify,
      }));
    } catch { /* private window, or storage disabled — the session still works */ }
  }

  function restoreAlerting() {
    try {
      const saved = JSON.parse(localStorage.getItem('ibvap.alerting') || '{}');
      // Sound stays off until the operator clicks: the browser needs a gesture
      // before it will play anything, so restoring it "on" would be a lie.
      state.alerting.sound = false;
      state.alerting.notify = Boolean(saved.notify)
        && Notifier.supported() && Notification.permission === 'granted';
      state.alerting.wantedSound = Boolean(saved.sound);
    } catch { /* ignore */ }
    renderAlertToggles();
  }

  function renderAlertToggles() {
    const sound = $('btn-sound');
    const notify = $('btn-notify');
    // Declared for the whole function: both buttons describe the same
    // escalation set, and scoping this to the Sound branch made the Notify
    // branch throw a ReferenceError that killed the rest of init().
    const red = state.alerting.redSeverities.join(' / ');
    if (sound) {
      sound.classList.toggle('armed', state.alerting.sound);
      sound.innerHTML = `<span class="btn-icon">${state.alerting.sound ? '🔊' : '🔇'}</span> Sound`;
      sound.title = state.alerting.sound
        ? `Audible alerts on for ${red} only. Click to mute.`
        : (state.alerting.wantedSound
            ? 'Audible alerts were on last time — click once to re-arm (browsers require a click before playing audio).'
            : `Siren on CRITICAL, alarm on HIGH, silent below. Routine detections never sound.`);
    }
    if (notify) {
      const blocked = Notifier.supported() && Notification.permission === 'denied';
      notify.classList.toggle('armed', state.alerting.notify);
      notify.classList.toggle('denied', blocked);
      notify.innerHTML = `<span class="btn-icon">${state.alerting.notify ? '🔔' : '🔕'}</span> Notify`;
      notify.title = blocked
        ? 'Blocked for this site — allow notifications in the browser\u2019s site settings.'
        : `Desktop notification for ${red} alerts only — intrusions, not routine `
          + 'detections — even when this tab is in the background.';
    }
  }

  function toast(message, kind = 'info', ms = 4200) {
    const el = document.createElement('div');
    el.className = `toast ${kind}`;
    el.textContent = message;
    $('toast-stack').appendChild(el);
    setTimeout(() => el.remove(), ms);
  }

  async function api(path, options = {}) {
    const response = await fetch(path, options);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const body = await response.json();
        if (body.detail) detail = body.detail;
      } catch { /* non-JSON error body */ }
      throw new Error(detail);
    }
    const type = response.headers.get('content-type') || '';
    return type.includes('application/json') ? response.json() : response.text();
  }

  const form = (obj) => {
    const fd = new FormData();
    Object.entries(obj).forEach(([k, v]) => {
      if (v !== null && v !== undefined && v !== '') fd.append(k, v);
    });
    return fd;
  };

  /* --------------------------------------------------------------- clock */
  // The header clock is IST regardless of where the browser thinks it is —
  // it is computed from the UTC epoch plus the fixed +05:30 offset, never
  // from the viewer's local timezone.
  function tickClock() {
    const now = new Date();
    const ist = new Date(now.getTime() + (330 + now.getTimezoneOffset()) * 60000);
    const hh = ist.getHours() % 12 || 12;
    const pad = (n) => String(n).padStart(2, '0');
    const ampm = ist.getHours() >= 12 ? 'PM' : 'AM';
    $('clock').textContent =
      `${pad(hh)}:${pad(ist.getMinutes())}:${pad(ist.getSeconds())} ${ampm} IST`;
    $('clock-date').textContent = ist.toLocaleDateString('en-IN',
      { day: '2-digit', month: 'short', year: 'numeric' }) + ' · Asia/Kolkata';
  }

  /* ---------------------------------------------------------- navigation */

  function showView(name) {
    state.view = name;
    document.querySelectorAll('.view').forEach((el) =>
      el.classList.toggle('active', el.dataset.view === name));
    document.querySelectorAll('[data-view]').forEach((el) => {
      if (el.tagName === 'BUTTON') el.classList.toggle('active', el.dataset.view === name);
    });
    if (name === 'events') loadEventLog();
    if (name === 'system') loadSystem();
    if (name === 'analyze') loadAnalysisHistory();
  }

  /* ------------------------------------------------------------- cameras */

  async function loadCameras() {
    try {
      const cameras = await api('/api/cameras');
      state.cameras = cameras;
      renderCameras();
      populateCameraSelects();
    } catch (err) {
      toast(`Could not load cameras: ${err.message}`, 'err');
    }
  }

  function renderCameras() {
    const grid = $('camera-grid');
    if (!state.cameras.length) {
      grid.innerHTML =
        '<div class="placeholder"><p>No cameras registered.</p>' +
        '<p class="muted">Add an RTSP/HTTP stream, a webcam index, or a video file.</p></div>';
      state.tiles.forEach((t) => t.destroy?.());
      state.tiles.clear();
      return;
    }

    const seen = new Set();
    state.cameras.forEach((cam) => {
      seen.add(cam.id);
      if (!state.tiles.has(cam.id)) grid.appendChild(buildTile(cam));
    });

    // Remove tiles for cameras that no longer exist.
    [...state.tiles.keys()].forEach((id) => {
      if (!seen.has(id)) {
        const tile = state.tiles.get(id);
        state.tiles.delete(id);          // delete first: stops the retry loop
        tile.destroy?.();
      }
    });
    grid.querySelector('.placeholder')?.remove();
    // A new tile starts in 'idle' and is given its mode here, so the number of
    // open MJPEG connections is decided in exactly one place.
    rebalanceStreams();
  }

  /**
   * Hand the live-stream budget to the tiles that most deserve it.
   *
   * Preference goes to tiles actually on screen: an operator scrolled down to
   * cameras 7-9 wants those live, not the three at the top they cannot see.
   * Everything else falls back to snapshot refresh. Called whenever the set of
   * tiles changes or the operator scrolls.
   */
  function rebalanceStreams() {
    const tiles = [...state.tiles.entries()];
    if (!tiles.length) return;

    const viewportH = window.innerHeight || 1080;
    const scored = tiles.map(([id, tile]) => {
      let visible = 0;
      try {
        const r = tile.root.getBoundingClientRect();
        // Fraction of the tile inside the viewport, 0 when fully off screen.
        const overlap = Math.max(0, Math.min(r.bottom, viewportH) - Math.max(r.top, 0));
        visible = r.height > 0 ? overlap / r.height : 0;
      } catch { /* detached mid-rebalance */ }
      return { id, tile, visible };
    });

    // Most-visible first; ties keep grid order so the assignment is stable and
    // tiles do not flip between modes on every scroll tick.
    scored.sort((a, b) => b.visible - a.visible);

    // A paused tile holds a still and has deliberately released its
    // connection, so it is not in the running for the budget at all. Without
    // this, a scroll would re-open its stream behind the frozen picture and
    // take back the socket the pause had just handed to the rest of the page.
    let live = 0;
    scored.forEach((entry) => {
      try {
        if (entry.tile.isPaused) return;
        if (live < STREAM_BUDGET) { entry.tile.goLive(); live += 1; }
        else entry.tile.goSnapshot();
      } catch { /* tile destroyed while we were deciding */ }
    });
  }

  function buildTile(cam) {
    const root = document.createElement('div');
    root.className = 'camera-tile';
    root.innerHTML = `
      <div class="camera-video">
        <img alt="${esc(cam.name)} feed" decoding="async">
        <canvas class="camera-frozen" hidden></canvas>
        <div class="zoom-bar" hidden>
          <button class="zoom-btn zoom-out" title="Zoom out">&minus;</button>
          <span class="zoom-level">1.0x</span>
          <button class="zoom-btn zoom-in" title="Zoom in">+</button>
          <button class="zoom-btn zoom-reset" title="Fit">Fit</button>
        </div>
        <div class="camera-badge"><span class="dot"></span><span class="b-status">CONNECTING</span></div>
        <div class="camera-source-badge ${cam.is_file_source ? 'file' : ''}"
             title="${esc(cam.is_file_source
                          ? 'Recorded video file: ' + (cam.source_name || '')
                          : cam.url)}">${esc(cam.source_label || 'LIVE FEED')}</div>
        <div class="camera-night" hidden>NIGHT MODE</div>
      </div>
      <div class="camera-bar">
        <div>
          <div class="camera-name">${esc(cam.name)}</div>
          <div class="camera-loc">${esc(cam.location || 'Location not set')}</div>
        </div>
        <div class="camera-metrics">
          <span><b class="m-fps">0.0</b> fps</span>
          <span><b class="m-obj">0</b> obj</span>
          <span><b class="m-lat">—</b> ms</span>
        </div>
      </div>
      <div class="camera-playback">
        <button class="pb-btn pb-toggle" title="Pause the picture (analysis keeps running)">&#10073;&#10073;</button>
        <span class="pb-speed-wrap">
          <label class="pb-label">Speed</label>
          <select class="pb-speed select select-sm">
            <option value="0.5">0.5x</option>
            <option value="0.75">0.75x</option>
            <option value="1" selected>1x</option>
            <option value="1.5">1.5x</option>
            <option value="2">2x</option>
          </select>
        </span>
        <span class="pb-note"></span>
      </div>
      <div class="camera-tools">
        <button class="btn btn-sm">Draw Fence</button>
        <button class="btn btn-sm btn-ghost">Restart</button>
        <button class="btn btn-sm btn-ghost">Events</button>
        <button class="btn btn-sm btn-danger" style="margin-left:auto">Remove</button>
      </div>`;

    const img = root.querySelector('img');
    // Decode off the main thread where the browser supports it. An MJPEG
    // <img> is re-decoded on every frame, and doing that synchronously on the
    // main thread is a frame-time spike on the same thread that renders the
    // rest of the dashboard — visible as a stutter the moment a camera is
    // added. (`loading="lazy"` was also removed from the markup: deferring
    // the load of a live stream only delays the first frame.)
    img.decoding = 'async';
    // Reconnect with exponential backoff, and stop entirely once the camera is
    // gone from our state.
    //
    // This used to be `img.onerror = () => { img.src = ... }` with no delay at
    // all. An <img> pointed at a stream that errors immediately — a removed
    // camera returning 404, a source that will not open — fires onerror, which
    // assigns src, which errors again, as fast as the browser can loop. That is
    // a self-inflicted denial of service: the tab pins a core rendering nothing
    // while the server answers thousands of requests a second, which is the
    // "the site lags / removing a camera behaves unpredictably" symptom, seen
    // from the client side rather than the pipeline.
    let retryDelay = 1000;
    let retryTimer = null;
    let snapTimer = null;
    let mode = 'idle';                          // 'live' | 'snapshot' | 'idle'

    const stopSnapshots = () => {
      if (snapTimer) { clearInterval(snapTimer); snapTimer = null; }
    };

    /** Live MJPEG: one connection, held open for as long as the tile shows. */
    const goLive = () => {
      if (mode === 'live') return;
      stopSnapshots();
      mode = 'live';
      root.classList.remove('tile-snapshot');
      img.src = `/stream/${cam.id}?t=${Date.now()}`;
    };

    /**
     * Snapshot mode: one short request per refresh, so the connection is
     * returned to the pool between frames instead of being held forever.
     * Lower frame rate, but it costs no permanent socket.
     */
    const goSnapshot = () => {
      if (mode === 'snapshot') return;
      mode = 'snapshot';
      root.classList.add('tile-snapshot');
      img.src = '';                             // end any MJPEG response first
      const refresh = () => {
        if (!state.tiles.has(cam.id) || mode !== 'snapshot') return;
        img.src = `/api/cameras/${cam.id}/snapshot?t=${Date.now()}`;
      };
      refresh();
      stopSnapshots();
      snapTimer = setInterval(refresh, SNAPSHOT_INTERVAL_MS);
    };

    img.onload = () => { retryDelay = 1000; };
    img.onerror = () => {
      if (mode === 'snapshot') return;          // a missed still is not an outage
      if (mode === 'frozen' || paused) return;  // we closed this stream on purpose
      if (retryTimer) return;                   // one retry in flight at a time
      if (!state.tiles.has(cam.id)) return;     // camera removed — do not retry
      retryTimer = setTimeout(() => {
        retryTimer = null;
        if (state.tiles.has(cam.id) && mode === 'live') {
          img.src = `/stream/${cam.id}?t=${Date.now()}`;
        }
      }, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 30000);
    };

    /* ------------------------------------------------------- frozen frame */
    // Pausing copies the frame currently on screen into a canvas and shows
    // that instead of the <img>. Two reasons it is a canvas and not just a
    // stopped stream: an MJPEG <img> whose src is cleared goes blank rather
    // than holding its last frame, and a canvas can be zoomed and panned
    // without refetching anything.
    const frozen = root.querySelector('.camera-frozen');
    const fctx = frozen.getContext('2d');

    function freezeTile() {
      const w = img.naturalWidth || state.frame.width;
      const h = img.naturalHeight || state.frame.height;
      try {
        frozen.width = w;
        frozen.height = h;
        fctx.drawImage(img, 0, 0, w, h);
      } catch {
        return false;             // nothing decoded yet — leave the stream up
      }
      frozen.hidden = false;
      img.hidden = true;
      // Releasing the MJPEG connection while paused also hands a socket back
      // to the browser's per-origin pool, which the rest of the page needs.
      //
      // Leaving `mode` on 'live' here was a bug: clearing src aborts the
      // request, which fires img.onerror, whose reconnect timer then saw a
      // 'live' tile and re-opened the stream behind the frozen picture —
      // quietly taking back the very socket the pause had just released.
      stopSnapshots();
      mode = 'frozen';
      img.src = '';
      resetZoom();
      return true;
    }

    function unfreezeTile() {
      frozen.hidden = true;
      img.hidden = false;
      resetZoom();
      mode = 'idle';              // force the allocator to re-open a stream
      rebalanceStreams();
    }

    /* --------------------------------------------------------------- zoom */
    // Digital zoom on the held frame. Only meaningful while paused: the point
    // is to inspect a face, a plate or a figure at the treeline in the frame
    // the operator stopped on. Zoom is applied as a CSS transform on the
    // canvas, so panning costs no redraw and the pixels stay as sharp as the
    // source frame allows.
    let zoom = 1;
    let panX = 0;
    let panY = 0;

    function clampPan() {
      // Never let the picture be dragged off its own frame: at zoom z the
      // image overhangs by (z-1)/2 of its size in each direction.
      const limit = Math.max(0, (zoom - 1) / (2 * zoom)) * 100;
      panX = Math.max(-limit, Math.min(limit, panX));
      panY = Math.max(-limit, Math.min(limit, panY));
    }

    function paintZoom() {
      clampPan();
      frozen.style.transform =
        `scale(${zoom}) translate(${panX}%, ${panY}%)`;
      root.classList.toggle('tile-zoomed', zoom > 1);
      const badge = root.querySelector('.zoom-level');
      if (badge) badge.textContent = `${zoom.toFixed(1)}x`;
    }

    function setZoom(next, originX, originY) {
      const before = zoom;
      zoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, next));
      if (zoom === before) return;
      if (zoom === 1) { panX = 0; panY = 0; }
      else if (originX != null) {
        // Keep the point under the cursor roughly still as we scale, which is
        // what makes wheel-zoom feel like magnifying rather than sliding.
        const shift = (1 / before - 1 / zoom) * 50;
        panX -= (originX - 0.5) * 2 * shift;
        panY -= (originY - 0.5) * 2 * shift;
      }
      paintZoom();
    }

    function resetZoom() { zoom = 1; panX = 0; panY = 0; paintZoom(); }

    frozen.addEventListener('wheel', (e) => {
      if (!paused) return;
      e.preventDefault();
      const r = frozen.getBoundingClientRect();
      setZoom(zoom * (e.deltaY < 0 ? 1.25 : 0.8),
              (e.clientX - r.left) / r.width, (e.clientY - r.top) / r.height);
    }, { passive: false });

    frozen.addEventListener('dblclick', () => {
      if (paused) setZoom(zoom > 1 ? 1 : 2);
    });

    // Drag to pan, in the frame's own percentage space so it is independent of
    // how large the tile happens to be rendered.
    let dragging = null;
    frozen.addEventListener('pointerdown', (e) => {
      if (!paused || zoom <= 1) return;
      dragging = { x: e.clientX, y: e.clientY, panX, panY };
      frozen.setPointerCapture(e.pointerId);
    });
    frozen.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      const r = frozen.getBoundingClientRect();
      panX = dragging.panX + ((e.clientX - dragging.x) / r.width) * 100;
      panY = dragging.panY + ((e.clientY - dragging.y) / r.height) * 100;
      paintZoom();
    });
    const endDrag = (e) => {
      if (!dragging) return;
      dragging = null;
      try { frozen.releasePointerCapture(e.pointerId); } catch { /* already gone */ }
    };
    frozen.addEventListener('pointerup', endDrag);
    frozen.addEventListener('pointercancel', endDrag);

    /* ---------------------------------------------------- playback control */
    // Pause is a *view* control. On a recording the server stops the decoder so
    // the footage waits; on a live camera it only holds the published picture
    // while capture, analytics and event sealing carry on — so freezing a tile
    // to look at something never blinds the post.
    const pbToggle = root.querySelector('.pb-toggle');
    const pbSpeed = root.querySelector('.pb-speed');
    const pbNote = root.querySelector('.pb-note');
    const pbWrap = root.querySelector('.pb-speed-wrap');
    let paused = false;

    const paintPlayback = () => {
      pbToggle.innerHTML = paused ? '&#9654;' : '&#10073;&#10073;';
      pbToggle.title = paused
        ? 'Resume the live picture'
        : 'Pause the picture (analysis keeps running)';
      pbToggle.classList.toggle('paused', paused);
      root.classList.toggle('tile-paused', paused);
      // Zoom only makes sense on a held frame, so its controls follow pause.
      const bar = root.querySelector('.zoom-bar');
      if (bar) bar.hidden = !paused;
    };

    root.querySelector('.zoom-in').onclick = () => setZoom(zoom * 1.5);
    root.querySelector('.zoom-out').onclick = () => setZoom(zoom / 1.5);
    root.querySelector('.zoom-reset').onclick = () => resetZoom();

    async function applyPlayback(body, optimistic) {
      try {
        const state = await api(`/api/cameras/${cam.id}/playback`, {
          method: 'POST', body: form(body),
        });
        paused = !!state.paused;
        if (state.speed) pbSpeed.value = String(state.speed);
        // A live camera cannot be played faster than it happens, so the server
        // reports whether speed means anything here rather than pretending.
        pbWrap.classList.toggle('unsupported', !state.speed_supported);
        pbSpeed.disabled = !state.speed_supported;
        pbNote.textContent = state.speed_supported
          ? (state.paused ? 'Paused' : `${state.effective_fps} fps`)
          : 'Live source — real time';
        paintPlayback();
        return state;
      } catch (err) {
        paused = optimistic;              // roll back to what the server has
        paintPlayback();
        toast(err.message, 'err');
        return null;
      }
    }

    pbToggle.onclick = () => {
      const want = !paused;
      paused = want;                      // optimistic, so the button feels instant
      paintPlayback();
      // Freeze what is on screen right now, so the held picture is the frame
      // the operator was looking at rather than whatever arrives next.
      if (want) freezeTile(); else unfreezeTile();
      applyPlayback({ paused: String(want) }, !want);
    };
    pbSpeed.onchange = () => applyPlayback({ speed: pbSpeed.value }, paused);
    paintPlayback();

    const [fence, restart, events, remove] = root.querySelectorAll('.camera-tools .btn');
    fence.onclick = () => openFence(cam.id, cam.name);
    restart.onclick = () => restartCamera(cam.id);
    events.onclick = () => { $('log-camera').value = cam.id; showView('events'); };
    remove.onclick = () => removeCamera(cam.id, cam.name);

    $('camera-grid').appendChild(root);
    state.tiles.set(cam.id, {
      root,
      // Detaching the element is not enough to end an MJPEG request: the
      // response never completes, so the browser can hold the connection (and
      // the server-side generator) open on a camera that no longer exists.
      // Clearing the handler first, then the src, is what actually closes it.
      destroy() {
        if (retryTimer) { clearTimeout(retryTimer); retryTimer = null; }
        stopSnapshots();
        mode = 'idle';
        img.onerror = null;
        img.onload = null;
        img.src = '';
        root.remove();
      },
      goLive,
      goSnapshot,
      get streamMode() { return mode; },
      get isPaused() { return paused; },
      dot: root.querySelector('.camera-badge .dot'),
      status: root.querySelector('.b-status'),
      night: root.querySelector('.camera-night'),
      fps: root.querySelector('.m-fps'),
      obj: root.querySelector('.m-obj'),
      lat: root.querySelector('.m-lat'),
    });
    return root;
  }

  /**
   * Hold the zone-occupancy signal for as long as a zone is occupied.
   *
   * Entering and leaving a polygon are moments, and the event feed already
   * records them. Being *inside* one is a condition, and a condition has to be
   * shown continuously or an operator who looks up thirty seconds after the
   * entry event sees an empty screen and assumes the area is clear. So this is
   * driven by state on the 1 Hz stats frame rather than by events: the banner
   * stays up, the tiles keep their ring, and the tone keeps pulsing, until the
   * backend says the polygon is empty again.
   */
  function renderZoneOccupancy(cameraStats) {
    let occupied = 0;
    let breached = 0;
    const names = [];

    cameraStats.forEach((s) => {
      const zones = s.zones || [];
      const camOccupied = zones.filter((z) => z.occupied);
      const camBreached = zones.filter((z) => z.breached);
      occupied += camOccupied.length;
      breached += camBreached.length;
      camOccupied.forEach((z) => {
        const seconds = Math.round(z.seconds || 0);
        names.push(`${s.name}·${z.rule}${seconds ? ` (${seconds}s)` : ''}`);
      });

      const tile = state.tiles.get(s.camera_id);
      if (tile) {
        tile.root.classList.toggle('zone-occupied', camOccupied.length > 0);
        tile.root.classList.toggle('zone-breached', camBreached.length > 0);
      }
    });

    state.zones.occupied = occupied;
    state.zones.breached = breached;

    const zonesStat = $('stat-zones');
    if (zonesStat) zonesStat.textContent = String(occupied);

    const banner = $('zone-banner');
    if (banner) {
      banner.hidden = occupied === 0;
      banner.classList.toggle('breached', breached > 0);
      if (occupied) {
        $('zone-banner-text').textContent = breached
          ? `INTRUSION IN PROGRESS — ${breached} zone${breached > 1 ? 's' : ''}`
          : `ZONE OCCUPIED — ${occupied} zone${occupied > 1 ? 's' : ''}`;
        $('zone-banner-detail').textContent = names.slice(0, 4).join('   ·   ')
          + (names.length > 4 ? `   · +${names.length - 4} more` : '');
      }
    }

    // The continuous audible signal. Sound.zonePulse rate-limits itself, so
    // calling it once a second produces a slow pulse, not a stutter.
    if (occupied) Sound.zonePulse(breached > 0);
  }

  function updateTiles(cameraStats) {
    renderZoneOccupancy(cameraStats);
    cameraStats.forEach((s) => {
      const tile = state.tiles.get(s.camera_id);
      if (!tile) return;
      // Show the lifecycle state, not just online/offline. "Connecting" and
      // "Error" used to render identically as a red OFFLINE dot, so an operator
      // could not tell a stream that needs a few more seconds from one whose
      // URL is wrong.
      // Every assignment below can invalidate layout, and this runs once a
      // second for every camera. Writing only what actually changed keeps a
      // multi-camera dashboard from doing a full style recalculation every
      // tick for values that are usually identical to last time.
      const colour = { green: 'dot-ok', amber: 'dot-warn', red: 'dot-down',
                       grey: 'dot-idle' }[s.state_colour] || 'dot-down';
      const dotClass = `dot ${colour}`;
      if (tile.dot.className !== dotClass) tile.dot.className = dotClass;

      const statusText = s.online ? 'LIVE' : (s.state_label || 'OFFLINE').toUpperCase();
      if (tile.status.textContent !== statusText) tile.status.textContent = statusText;
      const statusTitle = s.state_detail || '';
      if (tile.status.title !== statusTitle) tile.status.title = statusTitle;

      tile.root.classList.toggle('offline', !s.online);
      tile.night.hidden = !s.night_mode;
      if (s.night_mode && s.scene) {
        // Night mode is a measurement of this camera's view, so show what was
        // measured. The old build inferred it from the clock and had nothing to
        // show — and was wrong whenever the scene disagreed with the hour.
        tile.night.textContent =
          s.scene.source === 'infrared' ? 'NIGHT MODE · IR' : 'NIGHT MODE';
        tile.night.title =
          `Darkness score ${s.scene.darkness} (night at ${s.scene.enter_threshold})`
          + `\nMean brightness ${s.scene.mean_luma} / 255`
          + `\nDark pixels ${Math.round((s.scene.dark_fraction || 0) * 100)}%`
          + `\nReason: ${s.scene.source}`;
      }
      const fps = s.fps.toFixed(1);
      if (tile.fps.textContent !== fps) tile.fps.textContent = fps;
      const objects = String(s.detections);
      if (tile.obj.textContent !== objects) tile.obj.textContent = objects;
      const latency = String(Math.round(s.latency_ms));
      if (tile.lat.textContent !== latency) tile.lat.textContent = latency;
    });
  }

  function flashTile(cameraId) {
    const tile = state.tiles.get(cameraId);
    if (!tile) return;
    tile.root.classList.add('alerting');
    setTimeout(() => tile.root.classList.remove('alerting'), 3500);
  }

  async function restartCamera(id) {
    try {
      await api(`/api/cameras/${id}/restart`, { method: 'POST' });
      toast('Camera restarting…', 'ok');
      const tile = state.tiles.get(id);
      if (tile) {
        const img = tile.root.querySelector('img');
        setTimeout(() => { img.src = `/stream/${id}?t=${Date.now()}`; }, 900);
      }
    } catch (err) { toast(err.message, 'err'); }
  }

  async function removeCamera(id, name) {
    // Say what removal actually does. The stream, threads and zones go; sealed
    // events stay, because the alert log is a hash chain and deleting rows from
    // the middle of it would invalidate every later row.
    if (!confirm(
      `Remove camera "${name}"?\n\n` +
      'Its stream stops, its processing threads are released and its ' +
      'zones/tripwires are deleted.\nRecorded events and evidence are kept in ' +
      'the audit log.'
    )) return;
    try {
      const result = await api(`/api/cameras/${id}`, { method: 'DELETE' });
      const tile = state.tiles.get(id);
      state.tiles.delete(id);            // delete first: stops the retry loop
      tile?.destroy?.();
      toast(result.detail || `Removed ${name}`, 'ok', 6500);
      await loadCameras();
    } catch (err) {
      toast(`Could not remove ${name}: ${err.message}`, 'err', 7000);
    }
  }

  /* ------------------------------------------------ add camera (live / MP4) */

  function openCameraModal() {
    setCameraKind('live');
    openModal('modal-camera');
  }

  /**
   * Switch the Add Camera form between a network stream and an MP4 file.
   * Both produce a first-class camera; only the transport differs.
   */
  function setCameraKind(kind) {
    state.camera.kind = kind;
    document.querySelectorAll('#cam-kind .seg').forEach((seg) =>
      seg.classList.toggle('active', seg.dataset.kind === kind));
    $('cam-live-fields').hidden = kind !== 'live';
    $('cam-file-fields').hidden = kind !== 'file';
    $('cam-url').required = kind === 'live';
    $('cam-submit').textContent =
      kind === 'file' ? 'Upload & Add Video Source' : 'Add Camera';
  }

  function cameraFileChosen(event) { setCameraFile(event.target.files[0]); }

  function setCameraFile(file) {
    if (!file) return;
    if (!/\.mp4$/i.test(file.name)) {
      toast('Only .mp4 files are supported as a camera source.', 'warn');
      return;
    }
    state.camera.file = file;
    const label = $('cam-file-label');
    label.textContent = `${file.name} · ${(file.size / 1048576).toFixed(1)} MB`;
    label.classList.add('chosen');
    // Offer the filename as the camera name so the operator rarely has to type.
    if (!$('cam-name').value.trim()) {
      $('cam-name').value = file.name.replace(/\.mp4$/i, '').slice(0, 40).toUpperCase();
    }
  }

  async function submitCamera(event) {
    event.preventDefault();
    const name = $('cam-name').value.trim();
    const location = $('cam-location').value.trim();
    const button = $('cam-submit');
    const restore = button.textContent;

    try {
      button.disabled = true;
      let created;

      if (state.camera.kind === 'file') {
        if (!state.camera.file) {
          toast('Choose an MP4 file first.', 'warn');
          return;
        }
        button.textContent = 'Uploading…';
        // Multipart, not the JSON helper: the file is streamed to disk
        // server-side rather than buffered in memory.
        const body = new FormData();
        body.append('file', state.camera.file);
        body.append('name', name);
        body.append('location', location);
        body.append('is_active', 'true');
        created = await api('/api/cameras/upload', { method: 'POST', body });
        toast(`Video source added: ${created.name}`, 'ok', 6000);
      } else {
        const url = $('cam-url').value.trim();
        if (!url) { toast('Enter a stream URL or webcam index.', 'warn'); return; }
        created = await api('/api/cameras', {
          method: 'POST',
          body: form({ name, url, location, is_active: 'true' }),
        });
        toast(`Camera added: ${created.name}`, 'ok');
      }

      closeModal('modal-camera');
      event.target.reset();
      resetCameraForm();
      await loadCameras();
    } catch (err) {
      toast(err.message, 'err', 7000);
    } finally {
      button.disabled = false;
      button.textContent = restore;
    }
  }

  function resetCameraForm() {
    state.camera.file = null;
    const label = $('cam-file-label');
    if (label) {
      label.textContent = 'Drop an MP4 here or click to choose';
      label.classList.remove('chosen');
    }
    setCameraKind('live');
  }

  function populateCameraSelects() {
    const options = state.cameras
      .map((c) => `<option value="${c.id}">${esc(c.name)}</option>`).join('');
    const logSelect = $('log-camera');
    const current = logSelect.value;
    logSelect.innerHTML = `<option value="">All cameras</option>${options}`;
    logSelect.value = current;

    const uploadSelect = $('upload-camera');
    const chosen = uploadSelect.value;
    uploadSelect.innerHTML =
      `<option value="">No fence rules</option>${
        state.cameras.map((c) => `<option value="${c.id}">Apply ${esc(c.name)} rules</option>`).join('')}`;
    uploadSelect.value = chosen;
  }

  /* -------------------------------------------------------------- alerts */

  function pushAlert(alert) {
    state.alerts.unshift(alert);
    if (state.alerts.length > MAX_FEED) state.alerts.length = MAX_FEED;
    appendAlertToFeed(alert);              // O(1) DOM work, not a full rebuild
    flashTile(alert.camera_id);
    const knownType = state.alertTypes.has(alert.alert_type);
    state.alertTypes.add(alert.alert_type);
    // Rebuilding the filter dropdown re-lays-out the toolbar; only do it when
    // a genuinely new alert type has appeared, which is rare.
    if (!knownType) refreshTypeFilter();
    if (alert.severity === 'CRITICAL') {
      toast(`${alert.icon} ${alert.title} — ${alert.camera_name}`, 'err', 7000);
    }
    // Audible and desktop signals, so an operator who is not looking at this
    // tab still finds out. Both are no-ops unless armed.
    Sound.event(alert.severity);
    Notifier.show(alert);
  }

  /**
   * Rebuild the whole feed. Used when the filter changes or the feed is reset —
   * NOT on every incoming alert; see `appendAlertToFeed`.
   *
   * A full rebuild throws away up to MAX_FEED cards and builds them again, and
   * the browser must re-layout and repaint all of them. Doing that per alert is
   * what made the dashboard stutter the moment a webcam was added: a camera
   * pointed at a person emits `human_detected` several times a second, so the
   * page was tearing down and rebuilding sixty DOM nodes at that rate, on the
   * main thread, competing with the MJPEG decode for the same frame budget.
   */
  function renderAlertFeed(freshId = null) {
    const filter = $('alert-filter').value;
    const list = filter ? state.alerts.filter((a) => a.severity === filter) : state.alerts;
    const feed = $('alert-feed');
    bindFeedDelegation(feed);

    if (!list.length) {
      feed.innerHTML = '<p class="empty">Monitoring. No alerts yet.</p>';
      return;
    }
    feed.innerHTML = list.map((a) => alertCard(a, a.id === freshId)).join('');
  }

  /**
   * One click listener for the whole feed, attached once.
   *
   * The previous code attached a fresh `onclick` to every card on every render:
   * sixty closures created and sixty discarded per alert. Delegation means one
   * listener for the life of the page, and cards become plain markup that can
   * be inserted without any JavaScript bookkeeping.
   */
  function bindFeedDelegation(feed) {
    if (!feed || feed.dataset.delegated === '1') return;
    feed.dataset.delegated = '1';
    feed.addEventListener('click', (event) => {
      const card = event.target.closest('.alert-card');
      if (card && feed.contains(card)) openAlert(Number(card.dataset.id));
    });
  }

  /**
   * Add one alert to the top of the feed without touching the other cards.
   *
   * Cost is O(1) in DOM work instead of O(MAX_FEED), which is the difference
   * between a feed that keeps up with a busy camera and one that makes the
   * whole page judder. Inserts are coalesced into one animation frame so a
   * burst of alerts costs a single layout pass rather than one each.
   */
  let feedPending = [];
  let feedFlushQueued = false;

  function appendAlertToFeed(alert) {
    feedPending.push(alert);
    if (feedFlushQueued) return;
    feedFlushQueued = true;
    requestAnimationFrame(() => {
      feedFlushQueued = false;
      const batch = feedPending;
      feedPending = [];
      flushFeed(batch);
    });
  }

  function flushFeed(batch) {
    const feed = $('alert-feed');
    if (!feed) return;
    bindFeedDelegation(feed);

    const filter = $('alert-filter').value;
    const visible = batch.filter((a) => !filter || a.severity === filter);
    if (!visible.length) return;

    feed.querySelector('.empty')?.remove();

    // Newest first, so insert in reverse and prepend each.
    const fragment = document.createDocumentFragment();
    visible.forEach((a) => {
      const holder = document.createElement('div');
      holder.innerHTML = alertCard(a, true);
      const card = holder.firstElementChild;
      if (card) fragment.appendChild(card);
    });
    feed.prepend(fragment);

    // Trim the tail to the retention cap — removing nodes is cheap; rebuilding
    // the ones that stay is not.
    while (feed.children.length > MAX_FEED) feed.lastElementChild.remove();
  }

  function alertCard(a, fresh) {
    const isAi = a.analysis_kind === 'AI DETECTION';
    return `
      <div class="alert-card sev-${esc(a.severity)}${fresh ? ' fresh' : ''}" data-id="${a.id}">
        <div class="alert-top">
          <span class="alert-icon">${a.icon || '🔔'}</span>
          <span class="alert-title">${esc(a.title)}</span>
          <span class="alert-sev sev-chip-${esc(a.severity)}">${esc(a.severity)}</span>
        </div>
        <div class="alert-desc">${esc(a.description || '')}</div>
        <div class="alert-meta">
          <span><b>${esc(a.camera_name)}</b></span>
          ${a.track_id ? `<span>${esc((a.object_class || 'OBJ').toUpperCase())} #${a.track_id}</span>` : ''}
          <span>${esc(a.timestamp_ist)}</span>
          <span class="alert-kind ${isAi ? 'ai' : ''}">${esc(a.analysis_kind)}</span>
          ${a.has_clip ? '<span>🎬</span>' : ''}
        </div>
      </div>`;
  }

  function clearFeed() { state.alerts = []; renderAlertFeed(); }

  /**
   * The server was hard-reset: empty this tab to match.
   *
   * Reloading the camera list is what actually matters. Every tile holds an
   * <img> on an MJPEG stream, and after a reset those cameras are gone — so
   * each tile would sit in its reconnect backoff hitting a 404 forever. Letting
   * renderCameras() destroy them is what closes those connections.
   */
  function onSystemReset(info) {
    state.alerts = [];
    state.alertTypes.clear();
    state.logOffset = 0;
    state.logTotal = 0;
    renderZoneOccupancy([]);          // no cameras, so no zone can be occupied
    refreshTypeFilter();
    renderAlertFeed();
    loadCameras();
    if (state.view === 'events') loadEventLog();
    toast(info?.detail || 'System reset — all cameras and events cleared.',
          'ok', 7000);
  }

  function refreshTypeFilter() {
    const select = $('log-type');
    const current = select.value;
    const types = [...state.alertTypes].sort();
    select.innerHTML = `<option value="">All types</option>${
      types.map((t) => `<option value="${esc(t)}">${esc(t.replace(/_/g, ' '))}</option>`).join('')}`;
    select.value = current;
  }

  async function openAlert(id) {
    openModal('modal-alert');
    $('alert-detail').innerHTML = '<div class="placeholder"><div class="spinner"></div></div>';
    try {
      const a = await api(`/api/alerts/${id}`);
      $('alert-detail').innerHTML = alertDetail(a);
    } catch (err) {
      $('alert-detail').innerHTML = `<p class="empty">Could not load event: ${esc(err.message)}</p>`;
    }
  }

  function alertDetail(a) {
    const cell = (label, value) =>
      `<div class="detail-cell"><span>${label}</span><b>${esc(value)}</b></div>`;

    const evidence = [];
    if (a.has_snapshot) {
      evidence.push(`<figure><figcaption>Snapshot (annotated)</figcaption>
        <img src="${a.snapshot_url}" alt="Event snapshot"></figure>`);
    }
    if (a.has_clip) {
      evidence.push(`<figure><figcaption>Evidence clip (pre + post event)</figcaption>
        <video src="${a.clip_url}" controls preload="metadata"></video></figure>`);
    }
    // The pixels the registration was actually read from. Without them the
    // plate is an assertion; with them it is evidence an operator can check.
    if (a.plate) {
      evidence.push(`<figure class="plate-evidence">
        <figcaption>Number plate crop — read as
          <b>${esc(a.plate)}</b>${a.plate_confidence != null
            ? ` at ${Math.round(a.plate_confidence * 100)}% OCR confidence` : ''}
          ${a.plate_verified ? '· matches Indian plate format'
                             : '· format unverified'}</figcaption>
        <img src="/api/alerts/${a.id}/plate" alt="Number plate crop"
             onerror="this.closest('figure').querySelector('figcaption').insertAdjacentHTML('beforeend','<br><span class=&quot;muted&quot;>crop not retained</span>'); this.remove();"></figure>`);
    }

    return `
      <div class="detail-head">
        <span class="d-icon">${a.icon || '🔔'}</span>
        <div>
          <h2>${esc(a.title)}</h2>
          <p class="muted">${esc(a.description || '')}</p>
        </div>
        <span class="alert-sev sev-chip-${esc(a.severity)}" style="margin-left:auto">${esc(a.severity)}</span>
      </div>

      <div class="detail-grid">
        ${cell('Event ID', `#${a.id}`)}
        ${cell('Timestamp (IST)', a.timestamp_ist)}
        ${cell('Camera', a.camera_name)}
        ${cell('Location', a.camera_location || '—')}
        ${cell('Object', a.object_class ? `${a.object_class.toUpperCase()} #${a.track_id}` : '—')}
        ${cell('Confidence', a.confidence ? `${(a.confidence * 100).toFixed(1)}%` : '—')}
        ${cell('Produced by', a.analysis_kind)}
        ${cell('Rule', a.rule_name || '—')}
        ${a.plate ? cell('Number plate', a.plate) : ''}
        ${a.watchlist_name ? cell('Watchlist match', a.watchlist_name) : ''}
        ${cell('Source', a.source_type === 'upload' ? 'Uploaded video' : 'Live camera')}
      </div>

      ${evidence.length ? `<div class="detail-evidence">${evidence.join('')}</div>`
        : '<p class="muted" style="margin-bottom:12px">No evidence artefacts recorded for this event.</p>'}

      <h3 style="font-size:12px;color:var(--text-dim);margin-bottom:6px">Event detail</h3>
      <pre class="detail-json">${esc(JSON.stringify(a.details, null, 2))}</pre>

      <div class="chain-line">
        <div>Chain position sealed with SHA-256.</div>
        <div>prev: ${esc(a.prev_hash)}</div>
        <div>this: ${esc(a.hash)}</div>
        <div>UTC stored: ${esc(a.timestamp)}</div>
      </div>`;
  }

  /* ----------------------------------------------------------- event log */

  async function loadEventLog(reset = true) {
    if (reset) state.logOffset = 0;
    const params = new URLSearchParams({ limit: PAGE_SIZE, offset: state.logOffset });
    const add = (key, value) => { if (value) params.set(key, value); };
    add('camera_id', $('log-camera').value);
    add('alert_type', $('log-type').value);
    add('severity', $('log-severity').value);
    add('source_type', $('log-source').value);
    add('search', $('log-search').value.trim());

    try {
      const data = await api(`/api/alerts?${params}`);
      state.logTotal = data.total;
      renderEventLog(data.alerts);
      $('events-count').textContent = `— ${data.total} record${data.total === 1 ? '' : 's'}`;
      const from = data.total ? state.logOffset + 1 : 0;
      const to = Math.min(state.logOffset + PAGE_SIZE, data.total);
      $('log-range').textContent = `${from}–${to} of ${data.total}`;
      $('log-prev').disabled = state.logOffset === 0;
      $('log-next').disabled = to >= data.total;
    } catch (err) {
      $('log-body').innerHTML =
        `<tr><td colspan="11" class="empty">Could not load log: ${esc(err.message)}</td></tr>`;
    }
  }

  /**
   * The registration recorded against an event, if any.
   *
   * A plate read is the substance of an ANPR event, so it belongs in the row
   * rather than inside the details of the row. Unverified reads are marked,
   * because a registration that does not match the plate grammar is a reading
   * rather than an identification.
   */
  function plateCell(alert) {
    if (!alert.plate) return '—';
    const verified = alert.plate_verified;
    const confidence = alert.plate_confidence != null
      ? ` ${Math.round(alert.plate_confidence * 100)}%` : '';
    return `<span class="plate-chip${verified ? ' verified' : ''}"
      title="${esc(alert.plate)}${confidence}${verified ? ' · matches Indian plate format' : ' · format unverified'}"
      >${esc(alert.plate)}</span>`;
  }

  function renderEventLog(alerts) {
    const body = $('log-body');
    if (!alerts.length) {
      body.innerHTML = '<tr><td colspan="11" class="empty">No events match these filters.</td></tr>';
      return;
    }
    body.innerHTML = alerts.map((a) => `
      <tr data-id="${a.id}" style="cursor:pointer">
        <td class="col-id">#${a.id}</td>
        <td class="col-time">${esc(a.timestamp_ist)}</td>
        <td><span class="alert-sev sev-chip-${esc(a.severity)}">${esc(a.severity)}</span></td>
        <td>${a.icon || ''} ${esc(a.title)}</td>
        <td>${esc(a.camera_name)}</td>
        <td>${a.object_class ? esc(a.object_class.toUpperCase()) : '—'}</td>
        <td class="col-plate">${plateCell(a)}</td>
        <td class="col-id">${a.track_id ? `#${a.track_id}` : '—'}</td>
        <td><span class="alert-kind ${a.analysis_kind === 'AI DETECTION' ? 'ai' : ''}">${esc(a.analysis_kind)}</span></td>
        <td>${a.has_snapshot ? '📷' : ''}${a.has_clip ? ' 🎬' : ''}${a.plate ? ' 🔢' : ''}${!a.has_snapshot && !a.has_clip && !a.plate ? '—' : ''}</td>
        <td class="col-hash" title="${esc(a.hash)}">${esc((a.hash || '').slice(0, 10))}…</td>
      </tr>`).join('');
    body.querySelectorAll('tr[data-id]').forEach((row) => {
      row.onclick = () => openAlert(Number(row.dataset.id));
    });
  }

  /**
   * Download the filtered event log as a PDF.
   *
   * The same filters the operator is looking at, so the printout matches the
   * screen. Fetched as a blob rather than opened in a tab: the endpoint sends
   * Content-Disposition: attachment, and a plain navigation would leave the
   * dashboard — and its live streams — for a download the browser then throws
   * away the tab for.
   */
  async function exportEventLogPdf() {
    const button = $('btn-export-pdf');
    const original = button ? button.innerHTML : '';
    const params = new URLSearchParams({ limit: 1000 });
    const add = (key, value) => { if (value) params.set(key, value); };
    add('camera_id', $('log-camera').value);
    add('alert_type', $('log-type').value);
    add('severity', $('log-severity').value);
    add('source_type', $('log-source').value);
    add('search', $('log-search').value.trim());

    if (button) { button.disabled = true; button.innerHTML = 'Building…'; }
    try {
      const response = await fetch(`/api/alerts/export.pdf?${params}`);
      if (!response.ok) {
        let detail = `${response.status} ${response.statusText}`;
        try { detail = (await response.json()).detail || detail; } catch { /* not JSON */ }
        throw new Error(detail);
      }
      const blob = await response.blob();
      const name = (response.headers.get('Content-Disposition') || '')
        .match(/filename="?([^"]+)"?/)?.[1] || 'ibvap-event-log.pdf';
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Revoke on the next turn: revoking synchronously can cancel the
      // download in some browsers before it has read the blob.
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      toast(`Event log saved as ${name}`, 'ok');
    } catch (err) {
      toast(`PDF export failed: ${err.message}`, 'err');
    } finally {
      if (button) { button.disabled = false; button.innerHTML = original; }
    }
  }

  function pageEvents(direction) {
    const next = state.logOffset + direction * PAGE_SIZE;
    if (next < 0 || next >= state.logTotal) return;
    state.logOffset = next;
    loadEventLog(false);
  }

  /* ------------------------------------------------------- virtual fence */

  const FENCE_MODES = {
    line:      { points: 2, hint: 'Click two points to place a tripwire across the scene.' },
    zone:      { points: 0, hint: 'Click at least 3 points to outline a restricted zone. Double-click to close it.' },
    loiter:    { points: 0, hint: 'Outline the area to watch for loitering. Double-click to close it.' },
    direction: { points: 2, hint: 'Click two points. Crossings against the permitted direction will alert.' },
  };

  async function openFence(cameraId, cameraName) {
    state.fence.cameraId = cameraId;
    state.fence.cameraName = cameraName;
    state.fence.points = [];
    $('fence-camera-name').textContent = cameraName;
    $('fence-name').value = '';
    openModal('modal-fence');
    setFenceMode('line');

    // Freeze a real frame from the camera to draw on, so the operator places
    // the fence against the actual scene rather than an empty canvas.
    const image = new Image();
    image.onload = () => { state.fence.image = image; drawFence(); };
    image.onerror = () => { state.fence.image = null; drawFence(); };
    image.src = `/api/cameras/${cameraId}/snapshot?t=${Date.now()}`;

    await loadFenceRules();
  }

  async function loadFenceRules() {
    try {
      state.fence.rules = await api(`/api/cameras/${state.fence.cameraId}/rules`);
    } catch { state.fence.rules = []; }
    renderFenceRules();
    drawFence();
  }

  function renderFenceRules() {
    const box = $('fence-rule-list');
    if (!state.fence.rules.length) {
      box.innerHTML = '<p class="empty">No rules on this camera.</p>';
      return;
    }
    box.innerHTML = state.fence.rules.map((r) => `
      <div class="rule-row">
        <span class="r-type">${esc(r.rule_type.toUpperCase())}</span>
        <div>
          <div class="r-name">${esc(r.name)}</div>
          <div class="r-geo">${r.geometry.length} point${r.geometry.length === 1 ? '' : 's'}
            ${r.is_active ? '· ARMED' : '· DISABLED'}</div>
        </div>
        <div class="r-actions">
          <button class="btn btn-sm btn-ghost" data-toggle="${r.id}">${r.is_active ? 'Disable' : 'Enable'}</button>
          <button class="btn btn-sm btn-danger" data-del="${r.id}">Delete</button>
        </div>
      </div>`).join('');

    box.querySelectorAll('[data-del]').forEach((btn) => {
      btn.onclick = () => deleteRule(Number(btn.dataset.del));
    });
    box.querySelectorAll('[data-toggle]').forEach((btn) => {
      btn.onclick = () => toggleRule(Number(btn.dataset.toggle));
    });
  }

  function setFenceMode(mode) {
    state.fence.mode = mode;
    state.fence.points = [];
    document.querySelectorAll('#fence-modes .seg').forEach((seg) =>
      seg.classList.toggle('active', seg.dataset.mode === mode));
    $('fence-hint').textContent = FENCE_MODES[mode].hint;

    const options = $('fence-options');
    if (mode === 'loiter') {
      options.innerHTML = `<label>Dwell threshold (seconds)
        <input class="input input-sm" id="opt-dwell" type="number" min="3" max="600" value="15"></label>`;
    } else if (mode === 'zone') {
      options.innerHTML = `<label>Presence alert after (seconds)
        <input class="input input-sm" id="opt-presence" type="number" min="1" max="600" value="5"></label>`;
    } else if (mode === 'direction') {
      options.innerHTML = `<label>Permitted direction
        <select class="select select-sm" id="opt-direction">
          <option value="entry">Entry (inbound)</option>
          <option value="exit">Exit (outbound)</option>
        </select></label>`;
    } else {
      options.innerHTML = '<span class="muted">Both crossing directions are reported (ENTRY / EXIT).</span>';
    }
    drawFence();
  }

  function canvasPoint(event) {
    const canvas = $('fence-canvas');
    const rect = canvas.getBoundingClientRect();
    // The canvas is displayed responsively; map the click back to the
    // analytics frame's own pixel space so the rule matches the video.
    return [
      Math.round((event.clientX - rect.left) * (canvas.width / rect.width)),
      Math.round((event.clientY - rect.top) * (canvas.height / rect.height)),
    ];
  }

  function onFenceClick(event) {
    const limit = FENCE_MODES[state.fence.mode].points;
    if (limit && state.fence.points.length >= limit) state.fence.points = [];

    const point = canvasPoint(event);
    // A double-click (used to close a polygon) also delivers two ordinary
    // click events at the same spot, which otherwise appended duplicate
    // vertices. Ignore a click that lands on top of the previous point.
    const last = state.fence.points[state.fence.points.length - 1];
    if (last && Math.hypot(point[0] - last[0], point[1] - last[1]) < 8) return;

    state.fence.points.push(point);
    drawFence();
    $('fence-hint').textContent = limit
      ? `${state.fence.points.length} of ${limit} points placed.`
      : `${state.fence.points.length} point(s) placed — double-click to close the shape.`;
  }

  function onFenceDouble() {
    if (FENCE_MODES[state.fence.mode].points === 0 && state.fence.points.length >= 3) {
      drawFence(true);
      $('fence-hint').textContent =
        `${state.fence.points.length} points placed — name it and press Save Rule.`;
    }
  }

  function drawFence(closed = false) {
    const canvas = $('fence-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state.fence.image) {
      ctx.drawImage(state.fence.image, 0, 0, canvas.width, canvas.height);
    } else {
      ctx.fillStyle = '#0a0d13';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#66718a';
      ctx.font = '13px system-ui';
      ctx.textAlign = 'center';
      ctx.fillText('Waiting for a frame from this camera…', canvas.width / 2, canvas.height / 2);
      ctx.textAlign = 'left';
    }

    // Existing armed rules, dimmed.
    state.fence.rules.forEach((rule) => {
      if (!rule.is_active || rule.geometry.length < 2) return;
      const isLine = rule.rule_type === 'line' || rule.rule_type === 'direction';
      strokeShape(ctx, rule.geometry, isLine ? '#00e2ff' : '#ff8a00', !isLine, 0.45);
    });

    // The rule currently being drawn.
    const points = state.fence.points;
    if (points.length) {
      const isLine = state.fence.mode === 'line' || state.fence.mode === 'direction';
      const isPolygon = !isLine && (closed || points.length >= 3);
      strokeShape(ctx, points, '#37b6ff', isPolygon, 1);
      points.forEach(([x, y], index) => {
        ctx.beginPath();
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = '#37b6ff';
        ctx.fill();
        ctx.strokeStyle = '#04121c';
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.fillStyle = '#e8edf5';
        ctx.font = 'bold 11px monospace';
        ctx.fillText(String(index + 1), x + 8, y - 8);
      });
    }
  }

  function strokeShape(ctx, points, colour, close, alpha) {
    if (points.length < 2) return;
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.beginPath();
    ctx.moveTo(points[0][0], points[0][1]);
    points.slice(1).forEach(([x, y]) => ctx.lineTo(x, y));
    if (close) ctx.closePath();
    ctx.strokeStyle = colour;
    ctx.lineWidth = 3;
    ctx.lineJoin = 'round';
    ctx.stroke();
    if (close) {
      ctx.globalAlpha = alpha * 0.18;
      ctx.fillStyle = colour;
      ctx.fill();
    }
    ctx.restore();
  }

  function resetFenceDraft() { state.fence.points = []; drawFence(); }

  async function saveFence() {
    const { mode, points, cameraId } = state.fence;
    const needed = mode === 'line' || mode === 'direction' ? 2 : 3;
    if (points.length < needed) {
      toast(`Place at least ${needed} points first.`, 'warn');
      return;
    }

    const params = {};
    if (mode === 'loiter') params.dwell_seconds = Number($('opt-dwell')?.value || 15);
    if (mode === 'zone') params.presence_seconds = Number($('opt-presence')?.value || 5);
    if (mode === 'direction') params.allowed_direction = $('opt-direction')?.value || 'entry';

    try {
      await api(`/api/cameras/${cameraId}/rules`, {
        method: 'POST',
        body: form({
          rule_type: mode,
          geometry: JSON.stringify(points),
          params: JSON.stringify(params),
          name: $('fence-name').value.trim(),
          is_active: 'true',
        }),
      });
      state.fence.points = [];
      $('fence-name').value = '';
      toast('Rule armed — it is now live on the camera.', 'ok');
      await loadFenceRules();
    } catch (err) { toast(err.message, 'err'); }
  }

  async function deleteRule(ruleId) {
    try {
      await api(`/api/rules/${ruleId}`, { method: 'DELETE' });
      toast('Rule removed', 'ok');
      await loadFenceRules();
    } catch (err) { toast(err.message, 'err'); }
  }

  async function toggleRule(ruleId) {
    const rule = state.fence.rules.find((r) => r.id === ruleId);
    if (!rule) return;
    try {
      await api(`/api/rules/${ruleId}`, {
        method: 'PUT',
        body: form({ is_active: rule.is_active ? 'false' : 'true' }),
      });
      await loadFenceRules();
    } catch (err) { toast(err.message, 'err'); }
  }

  /* ------------------------------------------------------- analyze video */

  function fileChosen(event) { setUploadFile(event.target.files[0]); }

  function setUploadFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.mp4')) {
      toast('Only .mp4 files are supported.', 'err');
      return;
    }
    state.upload.file = file;
    $('selected-file').textContent =
      `${file.name} · ${(file.size / (1024 * 1024)).toFixed(1)} MB`;
    $('btn-analyze').disabled = false;
  }

  async function startAnalysis() {
    const file = state.upload.file;
    if (!file) return;

    const body = new FormData();
    body.append('file', file);
    const cameraId = $('upload-camera').value;
    if (cameraId) { body.append('camera_id', cameraId); body.append('apply_rules', 'true'); }

    $('btn-analyze').disabled = true;
    $('btn-analyze').textContent = 'Uploading…';
    try {
      const result = await api('/api/analysis/upload', { method: 'POST', body });
      state.upload.sessionId = result.session_id;
      $('analysis-active').hidden = false;
      $('analysis-filename').textContent = result.filename;
      $('btn-download-analysis').hidden = true;
      $('btn-view-analysis-events').hidden = true;
      $('btn-cancel-analysis').hidden = false;
      toast(`Analysing ${result.filename} (${result.rules_applied} rule(s) applied)`, 'ok');
      startPreviewLoop();
    } catch (err) {
      toast(err.message, 'err');
    } finally {
      $('btn-analyze').disabled = false;
      $('btn-analyze').textContent = 'Analyze Video';
    }
  }

  function startPreviewLoop() {
    stopPreviewLoop();
    const img = $('analysis-preview');
    state.upload.poll = setInterval(() => {
      if (!state.upload.sessionId) return;
      img.src = `/api/analysis/${state.upload.sessionId}/preview?t=${Date.now()}`;
    }, 500);
  }

  function stopPreviewLoop() {
    if (state.upload.poll) { clearInterval(state.upload.poll); state.upload.poll = null; }
  }

  function onAnalysisProgress(data) {
    if (data.session_id !== state.upload.sessionId) { loadAnalysisHistory(); return; }

    $('analysis-progress-fill').style.width = `${data.progress}%`;
    $('analysis-progress-text').textContent = `${data.progress.toFixed(1)}%`;
    $('analysis-status').textContent = data.status;
    $('analysis-overlay').textContent = data.status.toUpperCase();

    const cell = (label, value) => `<div class="mini-stat"><span>${label}</span><b>${value}</b></div>`;
    $('analysis-stats').innerHTML =
      cell('Frames', `${data.analysed_frames}/${data.total_frames || '?'}`) +
      cell('Video time', `${data.current_time_seconds.toFixed(1)}s`) +
      cell('Speed', `${data.processing_fps.toFixed(1)} fps`) +
      cell('Detections', data.detections_total) +
      cell('Persons', data.persons) +
      cell('Vehicles', data.vehicles) +
      cell('Events', data.alerts) +
      cell('Resolution', data.resolution);

    if (['completed', 'failed', 'cancelled'].includes(data.status)) {
      stopPreviewLoop();
      $('btn-cancel-analysis').hidden = true;
      if (data.output_url) {
        const link = $('btn-download-analysis');
        link.href = data.output_url;
        link.hidden = false;
      }
      $('btn-view-analysis-events').hidden = false;
      if (data.status === 'completed') {
        toast(`Analysis complete — ${data.alerts} event(s) from ${data.analysed_frames} frames`, 'ok', 8000);
      } else if (data.status === 'failed') {
        toast(`Analysis failed: ${data.error}`, 'err', 9000);
      }
      loadAnalysisHistory();
    }
  }

  async function cancelAnalysis() {
    if (!state.upload.sessionId) return;
    try {
      await api(`/api/analysis/${state.upload.sessionId}/cancel`, { method: 'POST' });
      toast('Cancelling analysis…', 'warn');
    } catch (err) { toast(err.message, 'err'); }
  }

  function showAnalysisEvents() {
    if (!state.upload.sessionId) return;
    $('log-source').value = 'upload';
    showView('events');
  }

  async function loadAnalysisHistory() {
    try {
      const data = await api('/api/analysis');
      const rows = [...data.active, ...data.history];
      const box = $('analysis-history');
      if (!rows.length) { box.innerHTML = '<p class="empty">No videos analysed yet.</p>'; return; }
      box.innerHTML = rows.map((r) => `
        <div class="history-row">
          <span class="status-chip status-${esc(r.status)}">${esc(r.status.toUpperCase())}</span>
          <div>
            <div class="h-name">${esc(r.filename)}</div>
            <div class="h-meta">${esc(r.created_at_ist || '')} · ${esc(r.resolution || '')}
              · ${(r.duration_seconds || 0).toFixed(1)}s</div>
          </div>
          <div class="h-stats">
            <span>${r.analysed_frames} frames</span>
            <span>${r.persons ?? 0} persons</span>
            <span>${r.vehicles ?? 0} vehicles</span>
            <span><b>${r.alerts ?? 0}</b> events</span>
            ${r.output_url ? `<a class="btn btn-sm" href="${r.output_url}" target="_blank">Play</a>` : ''}
          </div>
        </div>`).join('');
    } catch { /* history is non-critical */ }
  }

  /* ------------------------------------------------------------- system */

  async function loadSystem() {
    try {
      const info = await api('/api/system/info');
      const d = info.detector || {};
      state.deviceLabel = d.device === 'cpu' ? 'CPU'
        : (d.gpu_name ? d.gpu_name.replace(/NVIDIA GeForce /, '') : (d.device || '—'));
      $('stat-device').textContent = state.deviceLabel;

      const card = (title, rows) => `
        <div class="sys-card"><h4>${title}</h4>${
          rows.map(([k, v, cls]) =>
            `<div class="sys-row"><span>${esc(k)}</span><b class="${cls || ''}">${esc(v)}</b></div>`).join('')
        }</div>`;

      $('system-grid').innerHTML =
        card('Inference', [
          ['Model', d.model || '—'],
          ['Device', d.device || '—', d.device && d.device !== 'cpu' ? 'good' : ''],
          ['GPU', d.gpu_name || 'not in use', d.cuda_available ? 'good' : 'bad'],
          ['FP16 half precision', d.half ? 'enabled' : 'disabled'],
          ['Inference size', `${d.imgsz}px`],
          ['Avg inference', `${d.inference_ms_avg} ms`],
          ['Frames inferred', d.inference_count],
        ]) +
        card('Pipeline', [
          ['Analytics resolution', info.pipeline.frame_size],
          ['Target FPS', info.pipeline.target_fps],
          ['Confidence threshold', info.pipeline.confidence],
          ['Alert cooldown', `${info.pipeline.debounce_seconds}s`],
          ['Night window (IST)', info.pipeline.night_hours_ist],
          ['System FPS', info.aggregate.system_fps],
          ['End-to-end latency', `${info.aggregate.avg_latency_ms} ms`],
        ]) +
        card('Face detection', [
          ['InsightFace', info.face.available ? 'available' : 'not installed',
            info.face.available ? 'good' : 'bad'],
          ['Detector size', `${info.face.det_size}px`],
          ['Runs every', `${info.face.cadence_frames} frames`],
          ['Watchlist entries', info.face.watchlist_count],
          ['Match threshold', info.face.threshold],
        ]) +
        card('ANPR', [
          ['EasyOCR', info.anpr.easyocr_installed ? 'available' : 'not installed',
            info.anpr.easyocr_installed ? 'good' : 'bad'],
          ['OCR device', info.anpr.device],
          ['Runs every', `${info.anpr.cadence_frames} frames (vehicles only)`],
          ['Avg OCR time', `${info.anpr.ocr_ms_avg} ms`],
          ['Plates read', info.anpr.plates_confident],
          ['Uncertain reads', info.anpr.plates_uncertain],
        ]) +
        card('Event log integrity', [
          ['Scheme', info.integrity.scheme],
          ['Total events', info.integrity.total_events],
          ['Merkle checkpoints', info.integrity.checkpoints],
          ['Chain tip', `${(info.integrity.chain_tip || '').slice(0, 24)}…`],
        ]) +
        card('Evidence store', [
          ['Snapshots', info.evidence.snapshots],
          ['Clips', info.evidence.clips],
          ['Processed videos', info.evidence.processed],
          ['Disk used', `${info.evidence.megabytes} MB / ${info.evidence.budget_mb} MB`],
        ]) +
        card('Runtime', [
          ['IBVAP version', info.version],
          ['Python', info.python],
          ['OpenCV', info.opencv],
          ['Timezone', info.timezone],
          ['Server time', info.server_time_ist],
          ['Dashboard clients', info.websocket_clients],
        ]);

      loadCheckpoints();
    } catch (err) {
      $('system-grid').innerHTML = `<p class="empty">Could not load system info: ${esc(err.message)}</p>`;
    }
  }

  async function loadCheckpoints() {
    try {
      const data = await api('/api/integrity/checkpoints?limit=10');
      const box = $('checkpoint-list');
      if (!data.checkpoints.length) {
        box.innerHTML = '<p class="muted" style="margin-top:10px">No checkpoints sealed yet.</p>';
        return;
      }
      box.innerHTML = `<h3 style="font-size:12px;color:var(--text-dim);margin:14px 0 8px">
          Merkle Checkpoints</h3>` +
        data.checkpoints.map((c) => `
          <div class="rule-row">
            <span class="r-type">SEALED</span>
            <div>
              <div class="r-name">Events #${c.first_alert_id}–#${c.last_alert_id} (${c.alert_count})</div>
              <div class="r-geo">${esc(c.timestamp_ist)} · root ${esc(c.merkle_root.slice(0, 20))}…</div>
            </div>
            <div class="r-actions">
              <button class="btn btn-sm btn-ghost" data-cp="${esc(c.checkpoint_uid)}">Verify</button>
            </div>
          </div>`).join('');
      box.querySelectorAll('[data-cp]').forEach((btn) => {
        btn.onclick = async () => {
          try {
            const result = await api(`/api/integrity/checkpoints/${btn.dataset.cp}/verify`);
            toast(result.message, result.valid ? 'ok' : 'err', 7000);
          } catch (err) { toast(err.message, 'err'); }
        };
      });
    } catch { /* checkpoints are supplementary */ }
  }

  /* ---------------------------------------------------------- integrity */

  async function verifyIntegrity() {
    const badge = $('integrity-badge');
    badge.className = 'badge badge-idle';
    badge.textContent = 'CHECKING…';
    try {
      const result = await api('/api/integrity/verify');
      badge.className = `badge ${result.valid ? 'badge-ok' : 'badge-fail'}`;
      badge.textContent = result.valid ? '✓ VERIFIED' : '✕ COMPROMISED';

      const out = $('integrity-output');
      out.className = `integrity-output ${result.valid ? 'ok' : 'fail'}`;
      out.textContent = [
        result.valid ? '✓  INTEGRITY VERIFIED' : '✕  INTEGRITY COMPROMISED',
        '',
        `Scheme          : ${result.scheme}`,
        `Events checked  : ${result.total_alerts}`,
        `Verified at     : ${result.verified_at_ist}`,
        `Walk duration   : ${result.duration_ms} ms`,
        `Chain tip       : ${result.chain_tip}`,
        '',
        result.message,
        result.broken_at ? `\nFirst broken record : #${result.broken_at}` : '',
        result.broken_at ? `Expected hash       : ${result.expected_hash}` : '',
        result.broken_at ? `Stored hash         : ${result.actual_hash}` : '',
      ].join('\n');

      toast(result.valid
        ? `✓ Chain intact — ${result.total_alerts} events verified in ${result.duration_ms} ms`
        : `✕ Tampering detected at event #${result.broken_at}`,
        result.valid ? 'ok' : 'err', 8000);
    } catch (err) {
      badge.className = 'badge badge-fail';
      badge.textContent = 'ERROR';
      toast(err.message, 'err');
    }
  }

  async function makeCheckpoint() {
    try {
      const result = await api('/api/integrity/checkpoint', { method: 'POST' });
      toast(result.created
        ? `Checkpoint sealed over ${result.checkpoint.alert_count} event(s)`
        : result.message, result.created ? 'ok' : 'warn');
      loadCheckpoints();
    } catch (err) { toast(err.message, 'err'); }
  }

  async function downloadCertificate() {
    try {
      const response = await fetch('/api/integrity/certificate',
        { method: 'POST', body: form({ issued_to: 'Evidentiary Review' }) });
      if (!response.ok) throw new Error('Certificate generation failed');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `ibvap_integrity_certificate.json`;
      link.click();
      URL.revokeObjectURL(url);
      toast('Integrity certificate exported', 'ok');
    } catch (err) { toast(err.message, 'err'); }
  }

  /* ---------------------------------------------------------- websocket */

  function connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const socket = new WebSocket(`${proto}://${location.host}/ws/alerts`);
    state.ws = socket;

    socket.onopen = () => {
      state.wsRetry = 0;
      setSystemStatus('ok', 'SYSTEM ONLINE');
    };

    socket.onmessage = (event) => {
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      switch (message.type) {
        case 'connected':
          (message.data.recent || []).forEach((m) => {
            if (m.type === 'alert') state.alerts.push(m.data);
          });
          state.alerts.reverse();
          state.alerts.forEach((a) => state.alertTypes.add(a.alert_type));
          refreshTypeFilter();
          renderAlertFeed();
          break;
        case 'alert':     pushAlert(message.data); break;
        case 'stats':     applyStats(message.data); break;
        case 'analysis':  onAnalysisProgress(message.data); break;
        case 'checkpoint':
          toast(`Merkle checkpoint sealed over ${message.data.alert_count} event(s)`, 'ok');
          break;
        case 'system_reset': onSystemReset(message.data); break;
      }
    };

    socket.onclose = () => {
      setSystemStatus('down', 'RECONNECTING');
      const delay = Math.min(1000 * 2 ** state.wsRetry, 15000);
      state.wsRetry += 1;
      setTimeout(connectWebSocket, delay);
    };

    socket.onerror = () => socket.close();
  }

  function setSystemStatus(kind, text) {
    $('system-status').querySelector('.dot').className = `dot dot-${kind}`;
    $('system-status-text').textContent = text;
  }

  function applyStats(data) {
    $('stat-cameras').innerHTML =
      `${data.cameras_online}<small>/${data.cameras_total}</small>`;
    $('stat-persons').textContent = data.persons_live;
    $('stat-vehicles').textContent = data.vehicles_live;
    $('stat-fps').textContent = data.system_fps.toFixed(1);
    $('stat-inference').innerHTML = `${data.avg_inference_ms}<small> ms</small>`;
    $('stat-latency').innerHTML = `${data.avg_latency_ms}<small> ms</small>`;
    updateTiles(data.cameras || []);
  }

  async function refreshTodayCount() {
    try {
      const stats = await api('/api/stats?hours=24');
      $('stat-today').textContent = stats.today;
    } catch { /* non-critical */ }
  }

  /* ------------------------------------------------------------- modals */

  function openModal(id)  { $(id).classList.add('open'); }
  function closeModal(id) {
    $(id).classList.remove('open');
    if (id === 'modal-fence') state.fence.points = [];
  }

  /* --------------------------------------------------------------- init */

  function init() {
    tickClock();
    setInterval(tickClock, 1000);
    restoreAlerting();

    connectWebSocket();
    loadCameras();
    refreshTodayCount();
    setInterval(refreshTodayCount, 20000);

    const canvas = $('fence-canvas');
    canvas.addEventListener('click', onFenceClick);
    canvas.addEventListener('dblclick', onFenceDouble);

    // Which tiles are on screen decides which ones get a live socket, so the
    // budget is re-cut when the operator scrolls or resizes. Coalesced into one
    // animation frame: a scroll fires these far faster than they matter.
    let rebalanceQueued = false;
    const queueRebalance = () => {
      if (rebalanceQueued) return;
      rebalanceQueued = true;
      requestAnimationFrame(() => { rebalanceQueued = false; rebalanceStreams(); });
    };
    window.addEventListener('scroll', queueRebalance, { passive: true });
    window.addEventListener('resize', queueRebalance);

    // Drag-and-drop for both the offline-analysis zone and the Add Camera
    // video-source zone. Same behaviour, so it is wired once.
    const wireDropzone = (element, onFile) => {
      if (!element) return;
      ['dragenter', 'dragover'].forEach((type) =>
        element.addEventListener(type, (e) => {
          e.preventDefault(); element.classList.add('dragging');
        }));
      ['dragleave', 'drop'].forEach((type) =>
        element.addEventListener(type, (e) => {
          e.preventDefault(); element.classList.remove('dragging');
        }));
      element.addEventListener('drop', (e) => onFile(e.dataTransfer.files[0]));
    };
    wireDropzone($('drop-zone'), setUploadFile);
    wireDropzone($('cam-dropzone'), setCameraFile);

    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') {
        document.querySelectorAll('.modal.open').forEach((m) => m.classList.remove('open'));
      }
    });

    api('/api/system/info').then((info) => {
      // Adopt the server's escalation policy rather than keeping a second copy.
      const policy = info.alerting || {};
      if (Array.isArray(policy.red_alert_severities) && policy.red_alert_severities.length) {
        state.alerting.redSeverities = policy.red_alert_severities;
      }
      if (policy.severity_order) state.alerting.severityOrder = policy.severity_order;
      if (policy.notify_min_severity) state.alerting.minSeverity = policy.notify_min_severity;
      renderAlertToggles();

      const limits = info.limits || {};
      $('upload-limit').textContent = limits.upload_max_mb ?? 512;

      const d = info.detector || {};
      $('stat-device').textContent = d.device === 'cpu' ? 'CPU'
        : (d.gpu_name ? d.gpu_name.replace(/NVIDIA GeForce /, '') : '—');

      // The fence canvas MUST match the analytics frame size exactly: the
      // coordinates saved for a rule are consumed verbatim by the rule engine.
      // Reading the size from the server removes the standing risk that
      // changing FRAME_WIDTH/FRAME_HEIGHT silently misplaces every zone and
      // tripwire drawn afterwards.
      const frame = info.frame || {};
      if (frame.width && frame.height) {
        state.frame = { width: frame.width, height: frame.height };
        const canvas = $('fence-canvas');
        if (canvas.width !== frame.width || canvas.height !== frame.height) {
          canvas.width = frame.width;
          canvas.height = frame.height;
          drawFence();
        }
      }
    }).catch(() => {});

    showView('operations');
  }

  document.addEventListener('DOMContentLoaded', init);

  return {
    showView, loadCameras, openCameraModal, submitCamera, restartCamera, removeCamera,
    toggleSound, toggleNotifications,
    setCameraKind, cameraFileChosen,
    renderAlertFeed, clearFeed, openAlert, loadEventLog, pageEvents, exportEventLogPdf,
    openFence, setFenceMode, saveFence, resetFenceDraft, deleteRule, toggleRule,
    fileChosen, startAnalysis, cancelAnalysis, showAnalysisEvents, loadAnalysisHistory,
    loadSystem, verifyIntegrity, makeCheckpoint, downloadCertificate,
    openModal, closeModal,
  };
})();
