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
  };

  /* ----------------------------------------------------------- utilities */

  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s ?? '').replace(/[&<>"']/g,
    (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

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
  }

  function buildTile(cam) {
    const root = document.createElement('div');
    root.className = 'camera-tile';
    root.innerHTML = `
      <div class="camera-video">
        <img alt="${esc(cam.name)} feed" loading="lazy">
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
      <div class="camera-tools">
        <button class="btn btn-sm">Draw Fence</button>
        <button class="btn btn-sm btn-ghost">Restart</button>
        <button class="btn btn-sm btn-ghost">Events</button>
        <button class="btn btn-sm btn-danger" style="margin-left:auto">Remove</button>
      </div>`;

    const img = root.querySelector('img');
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
    const connect = () => {
      img.src = `/stream/${cam.id}?t=${Date.now()}`;
    };
    img.onload = () => { retryDelay = 1000; };
    img.onerror = () => {
      if (retryTimer) return;                   // one retry in flight at a time
      if (!state.tiles.has(cam.id)) return;     // camera removed — do not retry
      retryTimer = setTimeout(() => {
        retryTimer = null;
        if (state.tiles.has(cam.id)) connect();
      }, retryDelay);
      retryDelay = Math.min(retryDelay * 2, 30000);
    };
    connect();

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
        img.onerror = null;
        img.onload = null;
        img.src = '';
        root.remove();
      },
      dot: root.querySelector('.camera-badge .dot'),
      status: root.querySelector('.b-status'),
      night: root.querySelector('.camera-night'),
      fps: root.querySelector('.m-fps'),
      obj: root.querySelector('.m-obj'),
      lat: root.querySelector('.m-lat'),
    });
    return root;
  }

  function updateTiles(cameraStats) {
    cameraStats.forEach((s) => {
      const tile = state.tiles.get(s.camera_id);
      if (!tile) return;
      // Show the lifecycle state, not just online/offline. "Connecting" and
      // "Error" used to render identically as a red OFFLINE dot, so an operator
      // could not tell a stream that needs a few more seconds from one whose
      // URL is wrong.
      const colour = { green: 'dot-ok', amber: 'dot-warn', red: 'dot-down',
                       grey: 'dot-idle' }[s.state_colour] || 'dot-down';
      tile.dot.className = `dot ${colour}`;
      tile.status.textContent = s.online
        ? 'LIVE'
        : (s.state_label || 'OFFLINE').toUpperCase();
      tile.status.title = s.state_detail || '';
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
      tile.fps.textContent = s.fps.toFixed(1);
      tile.obj.textContent = s.detections;
      tile.lat.textContent = Math.round(s.latency_ms);
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
    renderAlertFeed(alert.id);
    flashTile(alert.camera_id);
    state.alertTypes.add(alert.alert_type);
    refreshTypeFilter();
    if (alert.severity === 'CRITICAL') {
      toast(`${alert.icon} ${alert.title} — ${alert.camera_name}`, 'err', 7000);
    }
  }

  function renderAlertFeed(freshId = null) {
    const filter = $('alert-filter').value;
    const list = filter ? state.alerts.filter((a) => a.severity === filter) : state.alerts;
    const feed = $('alert-feed');

    if (!list.length) {
      feed.innerHTML = '<p class="empty">Monitoring. No alerts yet.</p>';
      return;
    }
    feed.innerHTML = list.map((a) => alertCard(a, a.id === freshId)).join('');
    feed.querySelectorAll('.alert-card').forEach((card) => {
      card.onclick = () => openAlert(Number(card.dataset.id));
    });
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
        `<tr><td colspan="10" class="empty">Could not load log: ${esc(err.message)}</td></tr>`;
    }
  }

  function renderEventLog(alerts) {
    const body = $('log-body');
    if (!alerts.length) {
      body.innerHTML = '<tr><td colspan="10" class="empty">No events match these filters.</td></tr>';
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
        <td class="col-id">${a.track_id ? `#${a.track_id}` : '—'}</td>
        <td><span class="alert-kind ${a.analysis_kind === 'AI DETECTION' ? 'ai' : ''}">${esc(a.analysis_kind)}</span></td>
        <td>${a.has_snapshot ? '📷' : ''}${a.has_clip ? ' 🎬' : ''}${!a.has_snapshot && !a.has_clip ? '—' : ''}</td>
        <td class="col-hash" title="${esc(a.hash)}">${esc((a.hash || '').slice(0, 10))}…</td>
      </tr>`).join('');
    body.querySelectorAll('tr[data-id]').forEach((row) => {
      row.onclick = () => openAlert(Number(row.dataset.id));
    });
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

    connectWebSocket();
    loadCameras();
    refreshTodayCount();
    setInterval(refreshTodayCount, 20000);

    const canvas = $('fence-canvas');
    canvas.addEventListener('click', onFenceClick);
    canvas.addEventListener('dblclick', onFenceDouble);

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
    setCameraKind, cameraFileChosen,
    renderAlertFeed, clearFeed, openAlert, loadEventLog, pageEvents,
    openFence, setFenceMode, saveFence, resetFenceDraft, deleteRule, toggleRule,
    fileChosen, startAnalysis, cancelAnalysis, showAnalysisEvents, loadAnalysisHistory,
    loadSystem, verifyIntegrity, makeCheckpoint, downloadCertificate,
    openModal, closeModal,
  };
})();
