# IBVAP — Intelligent Border Video Analytics Platform

**Smart India Hackathon 2026 · Problem Statement 26187 · Ministry of Home Affairs / Sashastra Seema Bal**

IBVAP is a software-defined video analytics platform that converts existing
CCTV infrastructure into an AI-powered border surveillance system. It performs
real-time object detection and tracking, virtual-fence and behavioural rule
evaluation, facial recognition, automatic number plate recognition, and
tamper-evident event logging — using only commodity compute, with no
proprietary smart cameras, FRS appliances or ANPR hardware.

Any frame source is supported: RTSP and HTTP network cameras, USB or
integrated webcams, mobile phones used as webcams, YouTube streams, and
recorded video files.

---

## Contents

- [Capabilities](#capabilities)
- [Technology stack](#technology-stack)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Dashboard](#dashboard)
- [Analytics reference](#analytics-reference)
- [Evidence and integrity](#evidence-and-integrity)
- [Alerting and escalation](#alerting-and-escalation)
- [Configuration](#configuration)
- [Command-line reference](#command-line-reference)
- [REST API reference](#rest-api-reference)
- [Testing](#testing)
- [Deployment](#deployment)
- [Project structure](#project-structure)
- [Performance](#performance)
- [License](#license)

---

## Capabilities

### Detection and tracking
- Multi-class object detection using YOLO11 with ByteTrack multi-object tracking
- Eight tracked classes: person, bicycle, car, motorcycle, bus, train, truck, boat
- Stable track identities across frames, enabling trajectory-based rules
- Automatic low-light enhancement (CLAHE) on dark frames

### Rule engine
| Rule | Geometry | Triggers on |
|---|---|---|
| Tripwire | Line | Directional crossing (entry / exit) |
| Restricted zone | Polygon | Entry, sustained presence, exit |
| Loiter zone | Polygon | Dwell beyond a configurable threshold |
| Direction rule | Directed line | Movement against the permitted direction |
| Night movement | Frame-wide | Sustained movement while the scene is dark |

### Recognition
- **Facial recognition** — SCRFD detection with ArcFace embeddings, cosine
  similarity matching against a persisted watchlist
- **ANPR** — plate localisation with EasyOCR text recognition, Indian plate
  grammar validation, and multi-frame consensus voting

### Operations
- Live multi-camera dashboard with MJPEG streaming
- Interactive on-image drawing of fences and zones
- Per-camera playback control: pause, resume, and 0.5x–2x review speed
- Digital zoom and pan on a paused frame
- Continuous zone-occupancy state, not just entry and exit events
- Offline analysis of uploaded video through the identical pipeline
- Event log with filtering, pagination and PDF export

### Evidence
- Automatic snapshot and video clip capture on alert
- SHA-256 hash-chained event log with Merkle checkpointing
- Exportable integrity certificates
- Retention sweeping under a configurable storage ceiling

### Integration
- REST API for pull-based integration with command-and-control systems
- WebSocket push for live alerts and 1 Hz runtime statistics
- MJPEG relay for video redistribution
- Outbound signed webhook and SMS escalation

---

## Technology stack

| Layer | Technology | Version |
|---|---|---|
| Runtime | Python | 3.11+ (3.14 supported) |
| Object detection | Ultralytics YOLO11 | 8.4+ |
| Tracking | ByteTrack | bundled with Ultralytics |
| Deep learning runtime | PyTorch (CUDA optional) | 2.x |
| Face detection / embedding | InsightFace (SCRFD + ArcFace) | 2.0 |
| Optical character recognition | EasyOCR | 1.7+ |
| Computer vision | OpenCV | 4.10+ |
| Numerics | NumPy | 2.x |
| Web framework | FastAPI | 0.115+ |
| ASGI server | Uvicorn | 0.30+ |
| ORM | SQLAlchemy | 2.0+ |
| Validation | Pydantic | 2.x |
| Database | SQLite (WAL mode) | bundled |
| PDF reporting | ReportLab | 4.0+ |
| Frontend | Vanilla HTML / CSS / JavaScript | no build step |
| Testing | pytest | 8.x |
| Containerisation | Docker, Docker Compose | — |

The frontend deliberately uses no framework, bundler or CDN dependency. The
dashboard is one HTML file, one stylesheet and one script, so it loads on an
isolated LAN with no internet access and requires no build pipeline.

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
   RTSP / HTTP ───► │            Capture thread                │
   Webcam      ───► │  VideoCapture → deque(maxlen=1)          │
   YouTube     ───► │  paced to the source's own frame rate    │
   Video file  ───► └────────────────┬─────────────────────────┘
                                     │ latest frame only
                    ┌────────────────▼─────────────────────────┐
                    │          Analytics thread                │
                    │  ┌────────────────────────────────────┐  │
                    │  │ YOLO11 + ByteTrack                 │  │
                    │  │ Scene illumination estimation      │  │
                    │  │ Rule engine (fence/zone/loiter/…)  │  │
                    │  │ Overlay rendering + JPEG encode    │  │
                    │  └────────────────────────────────────┘  │
                    └──────┬──────────────────┬────────────────┘
                           │                  │
            ┌──────────────▼───┐   ┌──────────▼──────────────┐
            │  FrameBuffer     │   │   EventManager          │
            │  (MJPEG source)  │   │   seals + hash-chains   │
            └──────────┬───────┘   └──────────┬──────────────┘
                       │                      │
                       │        ┌─────────────┼─────────────┐
                       │        │             │             │
                 ┌─────▼────────▼──┐  ┌───────▼───┐  ┌──────▼──────┐
                 │  FastAPI / ASGI │  │  SQLite   │  │  Escalation │
                 │  REST + WS      │  │  evidence │  │  alarm/SMS  │
                 └─────────┬───────┘  └───────────┘  └─────────────┘
                           │
                 ┌─────────▼────────┐
                 │    Dashboard     │
                 └──────────────────┘

   Optional enrichment stages (face recognition, ANPR) run beside the
   analytics thread under a shared, process-wide duty ceiling.
```

### Threading model

| Thread | Count | Responsibility |
|---|---|---|
| Capture | 1 per camera | Owns the `VideoCapture`, decodes, publishes the latest frame |
| Analytics | 1 per camera | Detection, tracking, rules, overlay, JPEG encode, publish |
| Enrichment | 1 shared per stage | Face and ANPR ticks, duty-limited process-wide |
| Escalation worker | 1 per channel | Network and GPIO dispatch for alarm and SMS |
| ASGI event loop | 1 | HTTP, WebSocket and MJPEG serving |

Every thread is created with `daemon=True`. The capture-to-analytics hand-off
is a single-slot deque, so a stall in inference causes frame drops rather than
unbounded memory growth.

### Data flow guarantees

- Each JPEG is encoded once per frame and shared by every viewer of that camera
- Detection runs on the analytics-resolution frame; ANPR and face recognition
  read from the original full-resolution frame when available
- Rule geometry is stored and evaluated in analytics frame coordinates
- All timestamps are stored in UTC and rendered in Indian Standard Time

---

## Installation

### Requirements

- Python 3.11 or newer
- ffmpeg available on the system PATH
- Approximately 3 GB of disk for model weights and dependencies
- Optional: NVIDIA GPU with CUDA 12.x for accelerated inference

### Install

```bash
git clone <repository-url>
cd SIH26187
pip install -r requirements.txt
python manage.py init
```

Model weights download automatically on first use.

### GPU acceleration

The default PyPI `torch` distribution is CPU-only. On a machine with an NVIDIA
GPU, install the CUDA build **before** installing the remaining requirements:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

Confirm the active device:

```bash
curl -s localhost:8000/api/system/info | python -m json.tool | grep -A3 detector
```

---

## Quick start

```bash
# Register a camera — webcam index, RTSP URL, YouTube link or file path
python manage.py camera-add --name "Gate 1" --url 0 --location "North Post"

# Start the server
python manage.py run
```

| Endpoint | URL |
|---|---|
| Dashboard | `http://localhost:8000/dashboard` |
| Interactive API docs | `http://localhost:8000/docs` |
| Health check | `http://localhost:8000/health` |

### Supported source formats

| Source | Example |
|---|---|
| USB / integrated webcam | `0`, `1`, `2` |
| Mobile phone as webcam | `1` (via DroidCam, Iriun, EpocCam) |
| RTSP camera | `rtsp://user:pass@192.168.1.64:554/Streaming/Channels/101` |
| HTTP / MJPEG camera | `http://192.168.1.80/video.cgi` |
| YouTube stream | `https://www.youtube.com/watch?v=...` |
| Recorded file | `samples/sample_border_scenario.mp4` |

---

## Dashboard

The dashboard is served at `/dashboard` and comprises four views.

### Operations

Live camera tiles with MJPEG video, per-camera frame rate, object count and
pipeline latency, and a lifecycle state that distinguishes connecting from
offline from processing.

**Playback controls** are available per tile:

| Control | Behaviour on a recording | Behaviour on a live camera |
|---|---|---|
| Pause / resume | Decoding stops; footage resumes where it was stopped | The displayed frame is held; capture, analytics and event sealing continue |
| Speed 0.5x–2x | Decoder rate is scaled accordingly | Not applicable; reported as unsupported |

Pausing a live camera never interrupts analysis or event recording. A paused
tile is labelled `PAUSED` and releases its video connection while held.

**Zoom** is available on a paused tile: 1x to 8x digital zoom with click-drag
panning, mouse-wheel zoom toward the cursor, double-click to toggle, and a
`Fit` control to reset.

**Connection budget.** Browsers permit six concurrent HTTP/1.1 connections per
origin. The dashboard streams at most four tiles live and refreshes the
remainder from periodic snapshots, labelled `SNAPSHOT`, reserving connections
for API traffic. Visible tiles are given streaming priority.

**Virtual fence editor.** `Draw Fence` opens a frozen frame from the selected
camera. Click to place vertices, double-click to close a polygon. The drawing
canvas is sized from the server's analytics frame dimensions, so drawn
geometry maps exactly onto the coordinate space the rule engine evaluates.

### Event log

Filterable by camera, event type, severity, source and free-text search, with
pagination. Each row links to its snapshot, clip and chain hash.

`PDF` exports the currently filtered log as a printable report containing the
applied filters, the number of events included against the number matched, a
severity and event-type summary, the full event table with chain hashes, and
the hash-chain verification status measured at the time of generation.

### Analyze video

Upload an MP4 for offline analysis through the identical pipeline. Produces a
session with its own event list and annotated output video.

### Looping video sources

A video file registered as a camera restarts when it reaches its end
(`LOOP_FILE_SOURCES`). Each restart is recorded as a `source_restarted` event,
so the seam between passes is visible in the log rather than implied.

Because a replay is identical footage, events already sealed on an earlier pass
are recognised as repeats and recorded once rather than once per lap
(`FILE_LOOP_SUPPRESS_REPEATS`). Anything genuinely new on a later pass is still
sealed. Disable the setting when a looping file stands in for a live feed and
every pass must be treated as fresh footage.

### System

Detector and device information, subsystem status, storage usage, escalation
channel state, and integrity controls — chain verification, checkpoint
sealing, and certificate download.

---

## Analytics reference

### Detection parameters

| Parameter | Default | Description |
|---|---|---|
| `MODEL_PATH` | `yolo11s.pt` | Detection model weights |
| `TARGET_FPS` | `15` | Analytics cadence |
| `FRAME_WIDTH` × `FRAME_HEIGHT` | `640` × `384` | Analytics resolution |
| `INFERENCE_IMGSZ` | `640` | Model input size |
| `DEFAULT_CONFIDENCE` | `0.25` | Minimum detection confidence |
| `MIN_OBJECT_AREA` | `400` px | Minimum detection area |

Model selection guidance:

| Model | Suitable for | Relative characteristics |
|---|---|---|
| `yolo11n.pt` | CPU-only edge hardware | Fastest, lowest small-object recall |
| `yolo11s.pt` | Default; GPU or capable CPU | Balanced accuracy and latency |
| `yolo11m.pt` | GPU hosts prioritising distant detection | Highest small-object recall |

### Crossing rules

A crossing is evaluated as a directional transition across a line.

| Parameter | Default | Description |
|---|---|---|
| `CROSSING_MIN_DISPLACEMENT` | `4.0` px | Minimum travel between observations |
| Dead band | `8.0` px | Zone around the line in which no decision is made |
| `CROSSING_MIN_TRACK_AGE` | `2` frames | Minimum track age before a crossing may fire |
| `CROSSING_MAX_GAP_SECONDS` | `2.0` s | Occlusion gap beyond which the trail restarts |
| `CROSSING_REARM_SECONDS` | `3.0` s | Per-direction re-arm interval |

### Zone rules

| Parameter | Default | Description |
|---|---|---|
| `ANCHOR_CONFIRMATION_FRAMES` | `3` | Consecutive frames inside before entry is confirmed |
| `ZONE_BOUNDARY_MARGIN` | `6.0` px | Boundary dead band |
| `ZONE_EXIT_GRACE_SECONDS` | `1.5` s | Continuous time outside before exit is declared |
| `ZONE_PRESENCE_SECONDS` | `5.0` s | Dwell before sustained presence is announced |
| `ZONE_PRESENCE_REPEAT_SECONDS` | `30.0` s | Re-announcement interval while occupied |

### Loiter rules

| Parameter | Default | Description |
|---|---|---|
| `LOITER_SECONDS` | `15.0` s | Default dwell threshold |
| `LOITER_EXIT_GRACE_SECONDS` | `3.0` s | Grace before dwell accumulation resets |
| `LOITER_REALERT_SECONDS` | `120.0` s | Re-alert interval for a continuing dwell |
| `LOITER_CLASSES` | `["person"]` | Classes subject to loiter evaluation |

### Night detection

Darkness is determined visually from the image rather than from the system
clock, so a floodlit compound at midnight is treated as lit and an unlit
culvert at noon is treated as dark.

Signals measured per frame on a downsampled copy:

| Signal | Purpose |
|---|---|
| `mean_luma` | Overall brightness |
| `dark_fraction` | Share of pixels below `NIGHT_DARK_PIXEL_VALUE`; robust to point light sources |
| `saturation`, `channel_spread` | Identify a monochrome sensor |
| `luma_std` | Scene structure; distinguishes a real view from a flat placeholder frame |

The fused darkness score is exponentially smoothed and latched: the state
changes only after `NIGHT_CONFIRM_FRAMES` consecutive dark frames or
`DAY_CONFIRM_FRAMES` consecutive bright frames, with separate enter and exit
thresholds providing hysteresis.

Infrared night-vision sources are identified separately, by a colourless image
that additionally carries scene structure and genuinely dark regions. Uniform
or low-saturation but well-lit frames — such as those produced by virtual
webcam drivers during connection — are classified as daylight.

### ANPR

Plate reading operates in four stages:

1. **Localisation** — candidate plate regions are located within each detected
   vehicle's lower region, searched at the source resolution when available
2. **Scheduling** — the OCR budget is allocated by rotation, prioritising
   vehicles that have waited longest and skipping vehicles whose plate is
   already confirmed, so coverage scales with the number of vehicles present
3. **Recognition** — EasyOCR reads upscaled candidate crops across several
   preprocessing variants
4. **Consensus** — readings are accumulated per vehicle track and resolved by
   exact agreement, falling back to per-character majority voting weighted by
   OCR confidence

| Parameter | Default | Description |
|---|---|---|
| `ANPR_EVERY_N_FRAMES` | `10` | Analytics frames between ANPR ticks |
| `ANPR_MAX_PLATES_PER_TICK` | `2` | OCR budget per tick |
| `ANPR_MIN_VOTES` | `2` | Readings required for consensus |
| `ANPR_CONFIDENCE_THRESHOLD` | `0.45` | Threshold for a confident read |
| `ANPR_ALERT_CONFIDENCE` | `0.55` | Threshold for logging a plate to the event log |
| `ANPR_UNVERIFIED_PENALTY` | `0.75` | Multiplier for reads not matching Indian plate grammar |
| `ANPR_CONSENSUS_BONUS` | `0.20` | Confidence bonus for multi-frame agreement |
| `ANPR_EDGE_URGENCY_FRACTION` | `0.18` | Frame margin within which a vehicle is treated as leaving |
| `ANPR_DEBLUR_ENABLED` | `true` | Attempt motion deconvolution on unreadable crops |
| `ANPR_DEBLUR_LENGTHS` | `(5, 9, 13)` | Smear lengths tried, in source pixels |
| `ANPR_DEBLUR_SNR` | `0.012` | Wiener noise-to-signal term |

**Plate validation.** A reading is accepted as format-verified only if it
matches the layout grammar *and* its two leading characters are a registration
code India actually issues. Layout alone is insufficient: `KH12DE1433` fits the
grammar perfectly, but no such state code exists, so the reading is known to be
wrong. Where exactly one legal code is a single character away, the code is
repaired; where several are equally close, the reading is left as the
recogniser produced it and reported as uncertain.

**Moving vehicles.** A vehicle crossing the frame smears its plate along the
direction of travel. Crops that ordinary preprocessing cannot read are retried
with Wiener deconvolution over a short ladder of smear lengths, and the
existing scorer keeps whichever reading wins. This runs only after the cheap
path has failed, so sharp footage incurs no additional cost. Vehicles near a
frame edge are given priority for the OCR budget, since they have the fewest
remaining opportunities to be read.

Reads below `ANPR_ALERT_CONFIDENCE` are displayed live as `PLATE UNCERTAIN`
and are not written to the evidentiary log as a registration number.

For footage outside India, set `ANPR_UNVERIFIED_PENALTY=1.0` so readings are
judged on recognition quality alone rather than against Indian plate grammar.

### Facial recognition

| Parameter | Default | Description |
|---|---|---|
| `FACE_DET_SIZE` | `320` | SCRFD input resolution |
| `FACE_MIN_HEIGHT` | `32` px | Minimum face height for recognition |
| `FACE_SIMILARITY_THRESHOLD` | `0.55` | Cosine similarity required for a watchlist match |
| `FACE_MATCH_CACHE_SECONDS` | `30.0` | Identity cache lifetime per track |

Watchlist entries are persisted to the database with precomputed embeddings
and reloaded at startup. Identity caching is keyed by camera and track so an
identity established on one camera is never attributed to another.

### Severity model

| Severity | Events | Escalates |
|---|---|---|
| `CRITICAL` | Inbound fence crossing, sustained restricted-zone presence, watchlist match | Siren, notification, banner |
| `HIGH` | Zone entry, loitering, wrong direction, night movement, camera offline | Alarm tone, notification |
| `MEDIUM` | Outbound fence crossing, subsystem degradation | Logged and displayed |
| `LOW` | Zone exit, plate read, face observed | Logged and displayed |
| `INFO` | Routine human and vehicle detection, video file restart | Logged only |

The escalation threshold is `NOTIFY_MIN_SEVERITY` (default `HIGH`) and is
served to the dashboard through `/api/system/info`.

---

## Evidence and integrity

### Capture

On alert, the system writes a JPEG snapshot and an MP4 clip spanning
`CLIP_PRE_SECONDS` before and `CLIP_POST_SECONDS` after the triggering moment
(4 seconds each by default), using a rolling pre-roll buffer.

ANPR events additionally record the registration on the event itself — shown in
the event log, the alert detail and the PDF report — together with the cropped
plate image the reading was taken from, retrievable at
`GET /api/alerts/{id}/plate`.

### Hash chain

Every event is sealed into a SHA-256 hash chain. Each record incorporates the
digest of its predecessor, so any modification, deletion or reordering
invalidates every subsequent link.

```bash
python manage.py integrity          # verify the chain
python manage.py checkpoint         # seal a Merkle checkpoint
```

```http
GET  /api/integrity/verify          # full verification
GET  /api/integrity/tip             # current chain head
POST /api/integrity/checkpoint      # seal a checkpoint
POST /api/integrity/certificate     # export a signed certificate
```

Verification reports every broken link, classified as a payload modification,
a fork, or a missing record.

### Retention

| Store | Bound | Reclamation |
|---|---|---|
| Application log | `LOG_MAX_MB` × (`LOG_BACKUP_COUNT` + 1) = 60 MB | Automatic rotation |
| Snapshots, clips, crops | `MAX_EVIDENCE_MB` = 2048 MB | Retention sweep |
| Per-track analytics state | `TRACK_STATE_TTL` = 120 s | Periodic garbage collection |
| In-memory event feed | 200 most recent | Ring buffer |
| Archived camera records | — | `python manage.py prune` |

Cameras that own sealed evidence are archived rather than deleted, preserving
referential integrity of the hash chain. Archived cameras are excluded from
all operational listings.

---

## Alerting and escalation

Two outbound channels carry `CRITICAL` and `HIGH` events off the dashboard.
Both are disabled by default.

```
EventManager.record()
      ├── WebSocket ─────────────► dashboard
      ├── AlarmManager.trigger ──┐  severity gate → cooldown → queue
      └── SMSNotifier.handle ────┘  → worker thread → webhook / GPIO / SMS
```

Dispatch is asynchronous. Gating and enqueueing occur on the analytics thread;
all network and GPIO work occurs on a dedicated bounded-queue worker, so a slow
or unreachable endpoint cannot stall a camera pipeline.

### Alarm configuration

```bash
ALARM_ENABLED=true
ALARM_WEBHOOK_URL=http://192.168.1.50/api/siren/trigger
ALARM_WEBHOOK_SECRET=<shared-secret>
ALARM_COOLDOWN_SECONDS=60.0

ALARM_GPIO_PIN=18                 # optional; requires RPi.GPIO
ALARM_GPIO_DURATION_SECONDS=5
ALARM_GPIO_ACTIVE_HIGH=false
```

Webhook payload:

```json
{
  "action": "trigger_alarm",
  "severity": "CRITICAL",
  "alert_type": "entry",
  "title": "INTRUSION — FENCE CROSSED (INBOUND)",
  "camera_name": "BOP-NORTH-01",
  "timestamp_ist": "12 Sep 2026 10:00:00 IST",
  "track_id": 7,
  "description": "Track 7 crossed the fence line heading inbound.",
  "details": {"rule_name": "north fence"}
}
```

When `ALARM_WEBHOOK_SECRET` is configured, requests carry an
`X-IBVAP-Signature` header containing the HMAC-SHA256 of the exact request
body. Receivers should verify it before actuating:

```python
expected = hmac.new(SECRET, request.body, hashlib.sha256).hexdigest()
if not hmac.compare_digest(request.headers["X-IBVAP-Signature"], expected):
    abort(401)
```

### SMS configuration

```bash
SMS_ENABLED=true
SMS_PROVIDER=twilio                           # or msg91
SMS_TO_NUMBER=+919876543210,+919000000000
SMS_COOLDOWN_SECONDS=120.0

TWILIO_ACCOUNT_SID=<sid>
TWILIO_AUTH_TOKEN=<token>
TWILIO_FROM_NUMBER=+12025550123

MSG91_AUTH_KEY=<key>
MSG91_TEMPLATE_ID=<dlt-template-id>
MSG91_SENDER_ID=IBVAPS
```

When both providers are configured, the non-primary provider is used as an
automatic fallback. MSG91 uses the v5 flow API against a registered DLT
template, as required for transactional SMS in India. Messages are folded to
ASCII and constrained to a single 160-character GSM-7 segment.

### Commissioning

```bash
curl http://localhost:8000/api/system/notifications
curl -X POST http://localhost:8000/api/system/alarm/test
curl -X POST http://localhost:8000/api/system/sms/test
```

Test endpoints bypass severity and cooldown gates and return `503` with a
diagnostic message when a channel is disabled or unconfigured.

### Degradation

| Condition | Behaviour |
|---|---|
| `twilio` not installed | MSG91 used; status reports `twilio_installed: false` |
| `RPi.GPIO` not installed | Webhook sink used; single warning logged |
| `requests` not installed | Falls back to `urllib` |
| Endpoint unreachable | Counted as failed and logged; pipeline unaffected |
| Nothing configured | Status reports `configured: false` |

---

## Configuration

All settings live in `core/config.py` and may be overridden through
environment variables or a `.env` file. 175 settings are available; the table
below covers those most commonly adjusted. See `.env.example` for the full
annotated set.

### Server and lifecycle

| Setting | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address |
| `PORT` | `8000` | Bind port |
| `AUTOSTART_CAMERAS` | `true` | Start registered cameras at boot |
| `FRESH_START` | `false` | Retire all cameras at boot |
| `LOOP_FILE_SOURCES` | `true` | Restart a video file when it reaches its end |
| `FILE_LOOP_SUPPRESS_REPEATS` | `true` | Record a replayed event once rather than once per pass |
| `MAX_STREAM_CLIENTS` | `12` | Concurrent MJPEG viewers per camera |
| `HARD_RESET_ENABLED` | `true` | Enable the destructive reset endpoint |
| `HARD_RESET_TOKEN` | — | Required header token for hard reset |
| `CORS_ORIGINS` | `*` | Permitted browser origins |
| `LOG_MAX_MB` / `LOG_BACKUP_COUNT` | `10` / `5` | Log rotation policy |

The repository ships a gitignored `.env` that sets `FRESH_START=true`, which is
the intended development and demonstration behaviour — the dashboard starts
empty on every run. Set it to `false` for a deployment so registered cameras
persist across restarts.

`FRESH_START` never deletes records that own sealed evidence; such cameras are
archived instead.

### Performance

| Setting | Default | Description |
|---|---|---|
| `STAGE_MAX_DUTY` | `0.25` | Process-wide duty ceiling for face and ANPR stages |
| `LIVE_CAPTURE_HEADROOM` | — | Pacing headroom for live network sources |
| `CAMERA_TIMEOUT` | — | Startup grace before a camera is declared offline |
| `LOOP_FILE_SOURCES` | `true` | Restart file sources on completion |
| `JPEG_QUALITY` | — | Encoding quality for streamed and stored frames |

The duty ceiling is shared across all cameras rather than allocated per
camera, so the cost of optional enrichment stages remains constant as cameras
are added.

---

## Command-line reference

```
python manage.py <command> [options]
```

| Command | Description |
|---|---|
| `init` | Create database tables |
| `seed` | Register a demonstration camera and tripwire |
| `cameras` | List cameras (`--all` includes archived) |
| `camera-add` | Register a camera (`--name`, `--url`, `--location`) |
| `camera-rm` | Retire a camera, preserving sealed evidence |
| `rules` | List rules for a camera |
| `rule-add` | Add a rule (`--rule-type`, `--geometry`, `--params`) |
| `rule-rm` | Delete a rule |
| `alerts` | List recent events |
| `stats` | Event and storage statistics |
| `integrity` | Verify the hash chain |
| `chain-repair` | Re-seal the chain after an authorised break |
| `checkpoint` | Seal a Merkle checkpoint |
| `sweep` | Run evidence retention immediately |
| `prune` | Reclaim archived camera records holding no evidence |
| `verify` | Audit every problem-statement capability against real footage |
| `reset` | Recreate the database schema (destructive) |
| `hard-reset` | Stop all pipelines and wipe all state (destructive) |
| `run` | Start the server (`--host`, `--port`, `--reload`, `--fresh`, `--no-autostart`) |

### Rule geometry examples

```bash
# Tripwire
python manage.py rule-add --camera-id 1 --rule-type line \
  --geometry "[[100,240],[540,240]]"

# Restricted zone
python manage.py rule-add --camera-id 1 --rule-type zone \
  --geometry "[[100,100],[500,100],[500,400],[100,400]]"

# Loiter zone with a 60-second threshold
python manage.py rule-add --camera-id 1 --rule-type loiter \
  --geometry "[[100,100],[500,100],[500,400],[100,400]]" \
  --params '{"dwell_seconds": 60}'

# One-way direction rule
python manage.py rule-add --camera-id 1 --rule-type direction \
  --geometry "[[100,240],[540,240]]" \
  --params '{"allowed_direction": "entry"}'
```

Coordinates are expressed in the analytics frame space
(`FRAME_WIDTH` × `FRAME_HEIGHT`, 640 × 384 by default). Changing the analytics
resolution invalidates previously stored geometry.

---

## REST API reference

Interactive documentation is available at `/docs`.

### System

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness, camera counts, subsystem state |
| `GET` | `/api/system/info` | Full system state, per-camera statistics, zone occupancy |
| `GET` | `/api/system/time` | Authoritative IST clock |
| `GET` | `/api/stats` | Event counts, storage usage, type breakdown |
| `POST` | `/api/system/evidence/sweep` | Run retention immediately |
| `POST` | `/api/system/hard-reset?confirm=true` | Wipe all state (destructive) |

### Cameras

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras` | List active cameras |
| `POST` | `/api/cameras` | Register a camera |
| `GET` | `/api/cameras/{id}` | Camera detail |
| `PUT` | `/api/cameras/{id}` | Update a camera |
| `DELETE` | `/api/cameras/{id}` | Retire a camera (idempotent) |
| `POST` | `/api/cameras/upload` | Register a camera from an uploaded file |
| `POST` | `/api/cameras/{id}/restart` | Restart a camera pipeline |
| `POST` | `/api/cameras/{id}/playback` | Set pause state and review speed |
| `GET` | `/api/cameras/{id}/snapshot` | Single JPEG frame |
| `GET` | `/stream/{id}` | MJPEG stream |

`POST /api/cameras/{id}/playback` accepts `paused` (boolean) and `speed`
(0.5–2.0) as form fields and returns the resulting state:

```json
{
  "camera_id": 12,
  "paused": false,
  "speed": 1.5,
  "speed_supported": true,
  "effective_fps": 18.0
}
```

### Rules

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/cameras/{id}/rules` | List rules for a camera |
| `POST` | `/api/cameras/{id}/rules` | Create a rule |
| `DELETE` | `/api/cameras/{id}/rules` | Delete all rules for a camera |
| `PUT` | `/api/rules/{id}` | Enable, disable or modify a rule |
| `DELETE` | `/api/rules/{id}` | Delete a rule |

### Events and evidence

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/alerts` | Filterable, paginated event log |
| `GET` | `/api/alerts/export.pdf` | Filtered event log as a PDF report |
| `GET` | `/api/alerts/{id}` | Event detail with explanation |
| `GET` | `/api/alerts/{id}/snapshot` | Event snapshot image |
| `GET` | `/api/alerts/{id}/clip` | Event video clip |
| `GET` | `/api/alerts/{id}/plate` | Cropped number plate recorded for the event |
| `WS` | `/ws/alerts` | Live alert and statistics stream |

Both `/api/alerts` and `/api/alerts/export.pdf` accept `camera_id`,
`alert_type`, `severity`, `track_id`, `source_type`, `session_id`, `search`,
`from_ts` and `to_ts`.

### Recognition

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/anpr/detections` | Plate reads |
| `GET` | `/api/anpr/detections/{id}/evidence` | Plate crop image |
| `GET` | `/api/anpr/status` | Reader state and metrics |
| `GET` | `/api/anpr/test/{camera_id}` | Diagnostic read against a live camera |
| `GET` | `/api/faces/detections` | Face observations |
| `GET` | `/api/faces/detections/{id}/evidence` | Face crop image |
| `GET` | `/api/face/test/{camera_id}` | Diagnostic detection against a live camera |
| `GET` | `/api/watchlist` | List watchlist entries |
| `POST` | `/api/watchlist` | Add a watchlist entry with a reference image |
| `DELETE` | `/api/watchlist/{id}` | Remove a watchlist entry |
| `PUT` | `/api/watchlist/threshold` | Adjust the match similarity threshold |

### Offline analysis

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/analysis/upload` | Submit a video for analysis |
| `GET` | `/api/analysis` | List analysis sessions |
| `GET` | `/api/analysis/{sid}` | Session status and progress |
| `GET` | `/api/analysis/{sid}/alerts` | Events produced by a session |
| `GET` | `/api/analysis/{sid}/video` | Annotated output video |
| `GET` | `/api/analysis/{sid}/preview` | Session preview frame |
| `POST` | `/api/analysis/{sid}/cancel` | Cancel a running session |

### Integrity

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/integrity/verify` | Verify the full chain |
| `GET` | `/api/integrity/tip` | Current chain head |
| `GET` | `/api/integrity/checkpoints` | List sealed checkpoints |
| `GET` | `/api/integrity/checkpoints/{uid}/verify` | Verify one checkpoint |
| `POST` | `/api/integrity/checkpoint` | Seal a new checkpoint |
| `POST` | `/api/integrity/certificate` | Export an integrity certificate |

### Escalation

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/system/notifications` | Combined channel status |
| `GET` | `/api/system/alarm/status` | Alarm channel status |
| `GET` | `/api/system/sms/status` | SMS channel status |
| `POST` | `/api/system/alarm/test` | Fire a test alarm |
| `POST` | `/api/system/sms/test` | Send a test SMS |

---

## Testing

```bash
python -m pytest tests/ -q            # full suite
python -m pytest tests/ -v            # verbose
python -m pytest tests/test_rules.py  # a single module
```

**421 tests**, organised by the behaviour they protect:

| Module | Cases | Coverage |
|---|---|---|
| `test_api.py` | 53 | HTTP surface, validation, edge-case inputs |
| `test_core.py` | 49 | Configuration, frame buffer, models, enhancement |
| `test_notifications.py` | 47 | Escalation gates, cooldowns, HMAC, SMS segmentation |
| `test_regressions.py` | 46 | Defect-specific regression coverage |
| `test_scenarios.py` | 40 | End-to-end pipeline runs over footage |
| `test_rules.py` | 36 | All rule types, geometry, confirmation, debounce |
| `test_alert_policy.py` | 22 | Severity grading and escalation policy |
| `test_lifecycle.py` | 21 | Camera start, stop and teardown ordering |
| `test_resources.py` | 14 | Thread, socket, session and memory guarantees |
| `test_occupancy.py` | 13 | Continuous zone occupancy |
| `test_startup_state.py` | 13 | Startup reconciliation and fresh-start behaviour |
| `test_anpr_coverage.py` | 12 | OCR scheduling, plate validation, motion deblur |
| `test_loop_replay.py` | 11 | Looping file seams and replay suppression |
| `test_playback.py` | 11 | Pause, resume and speed semantics |
| `test_dashboard_init.py` | 10 | Dashboard bundle initialisation and stream budget |
| `test_report_pdf.py` | 8 | PDF report structure and content |
| `test_plate_evidence.py` | 6 | Number plate as event evidence |
| `test_face_cache.py` | 5 | Identity cache lifetime and bounds |
| `test_anpr_reading.py` | 4 | Plate grammar and glyph disambiguation |

Counts are collected cases, including parametrised variants.

`test_dashboard_init.py` executes `static/js/app.js` under Node.js against a
minimal DOM implementation, verifying that dashboard initialisation completes
and that event handlers are bound. Node.js is optional; these tests skip when
it is unavailable.

### Capability audit

```bash
python manage.py verify                 # audit against bundled footage
python manage.py verify --webcam        # include live face detection
python manage.py verify --json report.json
```

The audit runs the real pipeline over real footage and reports measured output
per capability. Capabilities that recorded daylight footage cannot demonstrate
— close-range face detection and night-time movement — are reported as
warnings rather than passes; use `--webcam` for live face verification.

### End-to-end acceptance

```bash
python verify_system.py                 # against a running server
```

Exercises the live HTTP and WebSocket surface with no mocks.

---

## Deployment

### Docker

```bash
docker build -t ibvap .
docker compose up -d
```

The image is based on `python:3.11-slim`. The `alerts/`, `clips/`, `videos/`,
`static/` and `dashboard/` paths are mounted as volumes so evidence and
configuration survive image rebuilds.

### Production checklist

| Item | Recommendation |
|---|---|
| `FRESH_START` | `false` — preserve registered cameras across restarts |
| `AUTOSTART_CAMERAS` | `true` — restore cameras after an unattended reboot |
| `HARD_RESET_TOKEN` | Set, or disable with `HARD_RESET_ENABLED=false` |
| `CORS_ORIGINS` | Restrict to known dashboard origins |
| `MODEL_PATH` | Match to available compute |
| `MAX_EVIDENCE_MB` | Match to available storage |
| `NOTIFY_MIN_SEVERITY` | Tune to operational escalation policy |
| Database | Back up `alerts.db` alongside the evidence tree |

The hash chain spans the database and the evidence files; back both up
together so integrity verification remains meaningful after a restore.

---

## Project structure

```
SIH26187/
├── api/
│   ├── main.py               # HTTP and WebSocket endpoints
│   └── schemas.py            # Request and response models
├── core/
│   ├── config.py             # Settings
│   ├── models.py             # ORM models
│   ├── database.py           # Engine, session factory, pooling
│   ├── camera.py             # CameraProcessor, CameraManager, FrameBuffer
│   ├── video_source.py       # Capture threads, source opening, playback control
│   ├── sources.py            # Registration and retirement
│   ├── analytics.py          # Per-frame analytics pipeline
│   ├── analysis.py           # Offline video analysis
│   ├── detections.py         # Detection records
│   ├── events.py             # Event sealing and fan-out
│   ├── evidence.py           # Snapshots, clip writing, retention
│   ├── hashchain.py          # SHA-256 chain, checkpoints, certificates
│   ├── report.py             # PDF event-log reporting
│   ├── notify.py             # Escalation channel base class
│   ├── alarm.py              # Siren channel (webhook / GPIO)
│   ├── sms.py                # SMS channel (Twilio / MSG91)
│   ├── youtube.py            # YouTube stream resolution
│   └── timeutil.py           # IST time handling
├── cv/
│   ├── detector.py           # YOLO11 and ByteTrack
│   ├── rules.py              # Rule engine
│   ├── geometry.py           # Lines, polygons, crossings
│   ├── face.py               # SCRFD, ArcFace, watchlist
│   ├── anpr.py               # Plate localisation, OCR, consensus
│   ├── scene.py              # Illumination and night estimation
│   └── overlay.py            # Video overlay rendering
├── dashboard/
│   └── index.html            # Operator interface
├── static/
│   ├── css/style.css
│   └── js/app.js
├── tests/                    # 398 tests
│   └── js/                   # Node-based dashboard harness
├── manage.py                 # Command-line interface
├── verify_capabilities.py    # Capability audit
├── verify_system.py          # End-to-end acceptance
├── benchmark.py              # Model comparison
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## Performance

Measured on an NVIDIA RTX 4060 Laptop GPU at 640 × 384 analytics resolution.

### Single camera

| Metric | Value |
|---|---|
| Inference latency (median) | 18.7 ms |
| End-to-end frame latency (median) | 24.7 ms |
| End-to-end frame latency (p95) | 58.8 ms |
| Frame age at consumption (median) | 17 ms |
| Publish-to-browser latency (median) | 3.5 ms |

### Eight concurrent cameras

| Metric | Value |
|---|---|
| Cameras online | 8 / 8 |
| Aggregate throughput | 57–96 fps |
| `/health` latency | 6.0 ms median, 30 ms p95 |
| `/api/cameras` latency | 6.5 ms median, 30 ms p95 |
| Teardown of all cameras | 263 ms |

### Model comparison

200 frames of project footage, RTX 4060, 640 × 384:

| Model | Confidence | Detections / frame | Small objects | Mean confidence | p50 latency |
|---|---|---|---|---|---|
| `yolo11n` | 0.30 | 1.46 | 5 | 0.651 | 18.8 ms |
| `yolo11s` | 0.25 | 1.68 | 8 | 0.714 | 18.4 ms |
| `yolo11m` | 0.25 | 1.82 | 14 | 0.733 | 22.7 ms |

Small objects are detections with a bounding box below 32 × 32 pixels.

### Resource characteristics

| Property | Behaviour |
|---|---|
| Per-camera thread cost | Fully released on camera removal |
| Per-camera handle cost | Fully released on camera removal |
| Memory under inference stall | Bounded; frames are dropped, not queued |
| Repeated add and remove cycles | No accumulation of threads, handles or descriptors |
| Process termination | No orphaned processes or retained device handles |

Run `benchmark.py` to reproduce the model comparison on your own hardware.

---

## License

MIT. See [LICENSE](LICENSE).
