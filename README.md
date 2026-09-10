# IBVAP - Intelligent Border Video Analytics Platform

**SIH 2026 · Project 26187 · Ministry of Home Affairs / SSB**

A software layer that gives existing ordinary CCTV cameras AI-powered border surveillance capabilities — intrusion detection, vehicle classification, face recognition, and tamper-evident alert logging — at the cost of one edge computer per border outpost instead of ₹40,000–₹1,50,000 per smart camera.

---

## 🚀 Quick Start

### What You Need
- Python 3.11+
- ffmpeg (`sudo apt install ffmpeg`)
- pip packages from `requirements.txt`

### One-Command Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Download the AI model (one-time)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Initialize the database
python manage.py init

# Add a test camera (webcam or video file)
python manage.py camera-add --name "Demo Camera" --url 0 --location "Border Post"

# Start the system
python manage.py run
```

**Then open your browser:**
- Dashboard: `http://localhost:8000/dashboard`
- API Docs: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

---

## 🎯 What This System Does

### Real-Time Detection & Tracking
- **YOLO11n + ByteTrack** for object detection with stable tracking
- Detects persons, vehicles (car, truck, bus, motorcycle, bicycle)
- Tracks objects using unique IDs across frames

### Smart Rules Engine - No False Alarms!
The system has four types of intelligent rules:

1. **Virtual Fence** — Draws a line on camera; detects when objects cross from outside to inside (or vice versa)
2. **Zone Intrusion** — Draws a polygon area; alerts when objects enter restricted zones
3. **Loiter Detection** — Alerts when objects stay in a zone longer than configured time
4. **Direction Rules** — Enforces one-way traffic on directed lines (like a one-way fence)

### Why It Won't False Alarm
Each rule requires **5 consecutive frames** confirming the object's position before triggering an alert. This means:
- ✅ Real people crossing a fence → Alert (confirmed for 1 second at 5 FPS)
- ❌ Tree branches swaying → Ignored (moves in < 5 frames)
- ❌ Shadows passing through → Ignored (doesn't stay long enough)

### Explainable AI
Every alert comes with a detailed human-readable explanation explaining:
- What was detected
- Confidence level
- Environmental conditions (day/night)
- Suggested actions

### Evidence Collection
When an alert triggers:
- Automatically saves snapshot of the event
- Records a short video clip (3-5 seconds)
- Creates tamper-evident cryptographic hash chain for legal integrity

---

## 🛡️ Three Critical Features (Pitfall-Proof)

### 1. Camera Disconnect Handling
If a camera drops, the system:
- Marks the camera as offline in the dashboard
- Keeps processing all other cameras
- Auto-reconnects every 0.5 seconds
- No crashes, no downtime

**Tested:** Works with both IP cameras and webcams

### 2. False Alert Prevention  
The system requires 5 consecutive frames (1+ second) to confirm any rule trigger. This prevents:
- Tree branches
- Passing shadows  
- Transient occlusions
- Wind-blown objects

**Tested:** All rules now use anchor confirmation properly

### 3. Live Webcam Demo
Works instantly with your laptop camera:

```bash
# Add your default webcam for live demo
python manage.py camera-add --name "Demo" --url 0 --location "Presentation"

# The camera starts automatically and streams live
```

---

## 📋 Available Commands

| Command | What It Does | Example |
|---------|--------------|---------|
| `init` | Initialize database | `python manage.py init` |
| `seed` | Add demo cameras | `python manage.py seed` |
| `cameras` | List all cameras | `python manage.py cameras` |
| `camera-add` | Add new camera | `python manage.py camera-add --name "Gate1" --url 0` |
| `camera-rm` | Remove camera | `python manage.py camera-rm --camera-id 3` |
| `rules` | List rules | `python manage.py rules --camera-id 1` |
| `rule-add` | Add rule | See examples below |
| `integrity` | Verify hash chain | `python manage.py integrity` |
| `stats` | Show alert stats | `python manage.py stats` |
| `reset` | ⚠️ Delete all data | `python manage.py reset` |
| `run` | Start server | `python manage.py run` |

### Adding Rules - Examples

```bash
# Add a fence (line from 100,240 to 540,240)
python manage.py rule-add --camera-id 1 --rule-type line --geometry "[[100,240],[540,240]]"

# Add a zone (polygon)
python manage.py rule-add --camera-id 1 --rule-type zone --geometry "[[100,100],[500,100],[500,400],[100,400]]"

# Add loitering rule with 60-second dwell time
python manage.py rule-add --camera-id 1 --rule-type loiter --geometry "[[100,100],[500,100],[500,400],[100,400]]" --params '{"dwell_seconds": 60}'

# Add direction rule (one-way traffic: left-to-right only)
python manage.py rule-add --camera-id 1 --rule-type direction --geometry "[[100,240],[540,240]]" --params '{"allowed_direction": "entry"}'
```

---

## 🐳 Docker Deployment

```bash
docker build -t ibvap .
docker-compose up -d
```

---

## 📁 Project Structure

```
SIH26187/
├── core/                 # Database, config, camera pipeline
│   ├── config.py        # All settings (FPS, rules, thresholds)
│   ├── models.py        # Camera, Rule, Alert database tables
│   ├── camera.py        # CameraProcessor with offline handling
│   └── hashchain.py     # SHA-256 tamper-evident chain
├── cv/                   # Computer vision modules
│   ├── detector.py      # YOLO + ByteTrack detection
│   ├── rules.py         # Fence, Zone, Loiter, Direction rules
│   ├── face.py          # Face recognition (stretch)
│   └── anpr.py          # License plate reading (stretch)
├── api/                  # FastAPI REST + WebSocket
│   └── main.py          # All API endpoints
├── dashboard/            # Web UI
│   └── static/          # CSS, JS, images
├── tests/                # pytest test suite
├── manage.py            # CLI management tool
└── requirements.txt     # Python dependencies
```

---

## 🧪 Testing

All tests pass (28 tests):

```bash
source ibvap_env/bin/activate
python -m pytest tests/ -v
```

**Test Results:**
- ✅ Config defaults
- ✅ Hash chain integrity
- ✅ Frame buffer
- ✅ Low-light enhancement
- ✅ Database models
- ✅ Detector parsing
- ✅ Rules engine
- ✅ Face recognizer

---

## 🔍 API Endpoints

| Type | Endpoint | Description |
|------|----------|-------------|
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Register a camera |
| GET | `/api/cameras/{id}` | Get camera details |
| DELETE | `/api/cameras/{id}` | Remove a camera |
| GET | `/api/cameras/{id}/rules` | List rules for camera |
| POST | `/api/cameras/{id}/rules` | Add a rule |
| DELETE | `/api/rules/{id}` | Remove a rule |
| GET | `/api/alerts` | Get all alerts |
| GET | `/api/alerts/{id}` | Get specific alert |
| GET | `/api/alerts/{id}/snapshot` | Get alert image |
| GET | `/api/alerts/{id}/clip` | Get alert video |
| GET | `/stream/{id}` | MJPEG video stream |
| WS | `/ws/alerts` | Live alert feed |
| GET | `/health` | System health check |
| GET | `/api/system/info` | System info with camera status |
| GET | `/api/integrity/verify` | Verify hash chain |

---

## 🎨 Dashboard Features

- **Live camera tiles** with MJPEG streaming
- **Interactive drawing** for fence/zone creation
- **Real-time alerts** with WebSocket push
- **AI explanations** for each alert
- **Evidence viewer** for snapshots and clips
- **Filter by** camera, type, time range
- **Integrity verification** button

---

## 📊 Key Technical Details

### Detection Parameters
- **FPS**: 5 (configurable)
- **Confidence**: 25% minimum
- **Object area**: 300 pixels minimum
- **Low-light**: Auto CLAHE enhancement

### Rules Parameters
- **Anchor confirmation**: 5 consecutive frames required
- **Debounce**: 10 seconds between same alert
- **Loiter time**: 60 seconds default

### Database
- **SQLite** for local deployment
- **HAProxy** compatible for production
- **Tamper-evident logging** for legal evidence

---

## 🆕 Recent Changes (SIH Fixes)

### Camera Management
- Added `is_online` status tracking
- Automatic fallback for offline cameras
- Graceful degradation when streams fail

### Rules Engine
- All rules now require 5-frame anchor confirmation
- Fixed DirectionRule alert logic bug
- Consistent side tracking across all rules

### API
- Camera status in `/health` endpoint
- Better error handling
- Improved documentation

---

## 📞 Support

For the SIH presentation, focus on:
1. **Camera add command**: `python manage.py camera-add --url 0`
2. **Dashboard**: Shows live feed from webcam
3. **Rules**: Demonstrate fence crossing with event prevention

---

**License: MIT**  
**Built for SIH 2026 - Border Security Challenge**