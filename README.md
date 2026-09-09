# IBVAP — Intelligent Border Video Analytics Platform

> **SIH 2026 · PS 26187 · Ministry of Home Affairs / SSB**

A software layer that gives existing ordinary CCTV cameras AI-powered border surveillance capabilities — intrusion detection, vehicle classification, face recognition, and tamper-evident alert logging — at the cost of one edge computer per border outpost instead of ₹40,000–₹1,50,000 per smart camera.

[![Build](https://img.shields.io/badge/status-Active-blue)](#)
[![API Docs](https://img.shields.io/badge/docs-Swagger-00BFFF)](#)
[![License](https://img.shields.io/badge/license-MIT-green)](#)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DASHBOARD (HTML/JS)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Camera   │  │ Camera   │  │ Camera   │  │ Alert    │    │
│  │ Tile #1  │  │ Tile #2  │  │ Tile #3  │  │ Feed     │    │
│  │ MJPEG    │  │ MJPEG    │  │ MJPEG    │  │ WS       │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
└───────┼──────────────┼──────────────┼──────────────┼──────────┘
        │              │              │              │
┌───────▼──────────────▼──────────────▼──────────────▼──────────┐
│                    FASTAPI SERVER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │ Cameras  │  │ Rules    │  │ Alerts   │  │ WebSocket│    │
│  │ CRUD     │  │ CRUD     │  │ + Hash   │  │          │    │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │
└───────┬──────────────────┬──────────────────┬───────────────┘
        │                  │                  │
┌───────▼──────────┐  ┌─────▼──────────┐  ┌───▼────────────┐
│  CAMERA PROC     │  │  RULES ENGINE  │  │  HASH CHAIN    │
│  YOLO + ByteTrack│  │  Fence/Zone/   │  │  SHA-256       │
│  + CLAHE + Face  │  │  Loiter/Direction│ │  Integrity     │
└──────────────────┘  └───────────────┘  └────────────────┘
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- ffmpeg (`sudo apt install ffmpeg`)
- pip packages from `requirements.txt`

### Install
```bash
# Clone and install
cd SIH26187
pip install -r requirements.txt

# Download YOLO model
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Initialize database
python manage.py init

# Seed demo data
python manage.py seed
```

### Run
```bash
# Start server
python manage.py run
# → Dashboard: http://localhost:8000/dashboard
# → API Docs:  http://localhost:8000/docs

# Or demo mode (generates synthetic video + seeds data)
python run_demo.py
```

### CLI Commands
```bash
python manage.py init              # Initialize database
python manage.py seed              # Seed with demo cameras/rules
python manage.py cameras           # List all cameras
python manage.py camera-add --name "BOP-04" --url "0"
python manage.py camera-rm --camera-id 3
python manage.py rules --camera-id 1
python manage.py rule-add --camera-id 1 --rule-type line --geometry "[[100,240],[540,240]]"
python manage.py rule-rm --rule-id 2
python manage.py integrity         # Verify hash chain
python manage.py stats             # Show alert statistics
python manage.py reset             # ⚠️ Reset database
```

### Docker
```bash
docker build -t ibvap .
docker-compose up -d
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/cameras` | List all cameras |
| POST | `/api/cameras` | Register a camera |
| DELETE | `/api/cameras/{id}` | Remove a camera |
| GET | `/api/cameras/{id}/rules` | List rules for a camera |
| POST | `/api/cameras/{id}/rules` | Save a rule (fence/zone/loiter/direction) |
| DELETE | `/api/rules/{id}` | Remove a rule |
| GET | `/api/alerts` | Query alerts (filter by camera, type, time) |
| GET | `/api/alerts/{id}` | Single alert detail |
| GET | `/api/alerts/{id}/snapshot` | Download evidence image |
| GET | `/api/alerts/{id}/clip` | Download evidence video |
| GET | `/stream/{camera_id}` | MJPEG live feed |
| WS | `/ws/alerts` | Real-time alert push |
| GET | `/api/integrity/verify` | Verify hash chain integrity |
| GET | `/api/integrity/tip` | Get latest chain hash |
| GET | `/api/stats` | Alert statistics |
| GET | `/api/watchlist` | List watchlist entries |
| POST | `/api/watchlist` | Add face to watchlist (upload image) |
| DELETE | `/api/watchlist/{id}` | Remove watchlist entry |
| PUT | `/api/watchlist/threshold` | Set similarity threshold |
| GET | `/api/system/info` | System info (models, CUDA, etc.) |
| GET | `/health` | Health check |

---

## Key Features

### 🎯 Detection & Tracking
- **YOLO11n + ByteTrack** for real-time object detection and persistent IDs
- Tracks persons, vehicles (car, truck, bus, motorcycle) with stable IDs
- Foot-point tracking (bottom-center) for accurate fence crossing detection

### 🛡️ Rules Engine
- **Virtual Fence** — line crossing detection (entry/exit) using cross-product geometry
- **Zone Intrusion** — polygon-based area detection using Shapely
- **Loitering** — dwell-time monitoring with configurable thresholds
- **Wrong Direction** — directional enforcement on directed lines
- Per-track debouncing to prevent alert flooding

### 🌙 Low-Light Enhancement
- **CLAHE** on L-channel for night-time detection
- Automatic activation when frame luminance drops below threshold

### 🔗 Tamper-Evident Hash Chain
- Every alert is hashed with the previous alert's SHA-256 hash
- Full chain verification reports exactly which row was tampered
- Cryptographic proof of alert log integrity for court-martial use

### 👤 Face Recognition (Stretch)
- **InsightFace** (SCRFD + ArcFace) for face detection and embedding
- Watchlist matching with cosine similarity thresholding
- Honest accuracy disclosure — investigative lead, not positive ID

### 📊 Dashboard
- Live camera tiles with MJPEG streaming
- Real-time WebSocket alert feed
- Interactive fence/zone drawing canvas
- Click alerts for evidence clips and full metadata
- Integrity verification button
- Camera, type, and time-range filters

---

## Research Foundation
- YOLO: Redmon et al. (2016), Sapon et al. (2026)
- ByteTrack: Zhang et al. (ECCV 2022)
- FaceNet: Schroff et al. (2015), ArcFace: Deng et al. (CVPR 2019)
- Indian License Plate: Tanwar et al. (2021), Nadiminti et al. (BARC 2022)
- UCF-Crime: Sultani et al. (CVPR 2018)
- Zero-DCE: Guo et al. (CVPR 2020)

---

## Project Structure
```
SIH26187/
├── core/                  # Database, config, models, hash chain, camera pipeline
│   ├── config.py          # Pydantic settings
│   ├── database.py        # SQLAlchemy SQLite setup
│   ├── models.py          # Camera, Rule, Alert ORM models
│   ├── hashchain.py       # SHA-256 tamper-evident chain
│   ├── camera.py          # CameraProcessor, FrameBuffer, ClipWriter
│   └── __init__.py
├── cv/                    # Computer vision modules
│   ├── detector.py        # YOLO + ByteTrack detection/tracking
│   ├── rules.py           # Fence, Zone, Loiter, Direction rules
│   ├── face.py            # InsightFace SCRFD + ArcFace
│   ├── anpr.py            # License-plate recognition (stretch goal)
│   └── __init__.py
├── api/                   # FastAPI application
│   ├── main.py            # All REST + WebSocket endpoints
│   ├── schemas.py         # Pydantic request/response models
│   └── __init__.py
├── dashboard/             # Web dashboard
│   ├── index.html         # Dashboard UI
│   └── static/
│       ├── css/style.css  # Dark theme styles
│       └── js/app.js      # Real-time JS
├── tests/                 # Pytest test suite
│   ├── conftest.py
│   ├── test_core.py
│   └── test_advanced_features.py
├── samples/               # Committed sample input media
│   └── sample_border_scenario.mp4
├── docs/                  # Supplementary documentation
│   ├── DEPLOYMENT.md      # Docker / Compose / Kubernetes guides
│   ├── DEMONSTRATION_SCRIPT.md
│   ├── HOW_TO_RUN_AND_SHOWCASE.md
│   ├── progress/          # Feature-completion development log
│   └── reference/         # Problem statement, hackathon guide PDF
├── manage.py              # CLI management tool
├── run_demo.py            # Demo runner with synthetic video
├── run_e2e_demo.py        # End-to-end demo with sample video
├── run_hackathon_demo.py  # Scripted judging demonstration
├── validate_enhancements.py # Conceptual validation checks
├── Dockerfile             # Container image
├── Dockerfile.prod        # Production container image
├── docker-compose.yml     # Compose configuration
├── requirements.txt       # Python dependencies
├── .env.example           # Template for environment variables
├── .gitignore             # Excludes venv, models, runtime artifacts
├── LICENSE                # MIT
└── README.md              # This file
```

---

## Why This Project Stands Out

1. **Hash Chain Integrity** — Most teams build pure CV and ignore the Blockchain & Cybersecurity theme. IBVAP uses a cryptographic hash chain for tamper-evident alert logging, providing court-admissible evidence integrity.

2. **Complete Pipeline** — End-to-end from camera feed to dashboard with evidence clips, not just a notebook demo.

3. **Real-Time Streaming** — MJPEG with zero-copy frame buffer, WebSocket alerts — no WebRTC/HLS overhead.

4. **Interactive Configuration** — Fence/zone drawing canvas proves per-site configurability, not hardcoded demo behavior.

5. **Measurable Performance** — Built to benchmark FPS at 1, 2, 4 streams with specific numbers for "how many cameras can one device handle?"

6. **Honest Assessment** — Face recognition presented as investigative lead (not positive ID), ANPR acknowledged as stretch goal. This maturity wins credibility with a Home Ministry panel.

7. **LAN Deployment Ready** — Designed for phone-webcam cameras on a local network with `0.0.0.0` binding and proper firewall configuration.

---

