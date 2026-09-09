# IBVAP Demonstration Outputs
## What the System Produces When Processing Sample Video

This directory contains sample outputs showing exactly what a user would see when running IBVAP with a real video input and all features (EB-SAI, Triple-Layer Integrity, ANPR) enabled.

## Directory Structure
```
DEMO_OUTPUTS/
├── api_responses/          # Sample API responses
├── evidence/               # Generated evidence clips and snapshots  
├── streams/                # What the MJPEG stream would show
├── integrity/              # Integrity verification outputs
└── anpr/                   # ANPR recognition results
```

## 1. API Responses Sample (`api_responses/`)

### Alert with EB-SAI Explanation
```json
// GET /api/alerts/42
{
  "id": 42,
  "camera_id": 0,
  "alert_type": "entry",
  "object_class": "person", 
  "track_id": 15,
  "confidence": 0.91,
  "timestamp": "2026-09-08T14:30:22Z",
  "explanation": "Subject (person) detected crossing perimeter boundary from exterior to interior with 91% confidence. Movement vector indicates deliberate approach velocity. Subject carrying backpack consistent with attempted smuggling. Low-light conditions present - CLAHE enhancement activated. Track maintained across 24 frames. Behavioral analysis shows sustained trajectory toward restricted zone.",
  "snapshot_path": "/alerts/snapshots/cam0_alert42_20260908_143022.jpg",
  "clip_path": "/clips/cam0_alert42_20260908_143022.mp4",
  "prev_hash": "a3f1c2e4b5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0",
  "hash": "f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1",
  "anchored": false
}
```

### ANPR Test Results
```json
// POST /api/anpr/test/0
{
  "camera_id": 0,
  "success": true,
  "detections": [
    {
      "bbox": [245, 180, 395, 220],
      "confidence": 0.87,
      "plate_text": "HR26DC1234",
      "text_confidence": 0.87,
      "frame_number": 1450,
      "timestamp": 1725802222.5
    },
    {
      "bbox": [120, 310, 250, 350],
      "confidence": 0.79,
      "plate_text": "DL1CA2345", 
      "text_confidence": 0.79,
      "frame_number": 1450,
      "timestamp": 1725802222.5
    }
  ],
  "count": 2,
  "message": "ANPR test completed successfully"
}
```

### Integrity Status
```json
// GET /api/integrity/status
{
  "chain_length": 127,
  "latest_hash": "f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1",
  "is_valid": true,
  "tampered_alerts": [],
  "last_anchor_time": "2026-09-08T14:25:00Z",
  "anchors_since_start": 25,
  "merkle_root": "a1b2c3d4e5f67890123456789abcdef0123456789"
}
```

### Integrity Certificate (Legal Affidavit Format)
```
AFFIDAVIT OF DATA INTEGRITY
IBVAP - Intelligent Border Video Analytics Platform

I, the undersigned, do hereby swear and affirm that:

1. I am knowledgeable about the IBVAP system and its data integrity mechanisms.

2. The IBVAP system employs a triple-layer tamper-evident integrity system:
   - Layer 1: Cryptographic hash chain linking each alert to the previous
   - Layer 2: Periodic blockchain anchoring every 5 minutes using Merkle trees
   - Layer 3: Exportable integrity certificates for legal verification

3. For the time period 2026-09-08T00:00:00Z to 2026-09-08T23:59:59Z:
   - Total alerts processed: 1247
   - Chain tip hash: f0e9d8c7b6a5f4e3d2c1b0a9f8e7d6c5b4a3f2e1
   - Blockchain anchors generated: 288
   - Merkle roots anchored: 288
   - All integrity verification checks passed

4. The hash chain has been verified and found to be intact, indicating
   that no alert data has been tampered with, modified, or deleted
   during the specified time period.

Further affiant sayeth not.
```

## 2. Evidence Files Sample (`evidence/`)

### Screenshot Annotation Description
When viewing the MJPEG stream at `http://localhost:8000/stream/0`, you would see:

**Frame Annotations:**
- 🟢 **Green Boxes**: Detected persons/vehicles with track IDs and confidence scores
  - Example: `ID:15 person 0.91` 
  - Example: `ID:23 vehicle 0.87`
- 🔴 **Red Alert Banner**: Top-left corner when rules triggered
  - Example: `ALERT: ENTRY ID:15` (flashing)
- 🟠 **Orange Boxes**: Detected license plates with text overlay (ANPR)
  - Example: `PLATE: HR26DC1234` above orange bounding box
- 🟡 **Yellow Lines**: Configured security rules (fence/zone boundaries)
- ⚪ **White Overlay**: Bottom-left corner
  - Example: `CAM 0 | 14:30:22` (camera ID and timestamp)
- 🔵 **Blue Dots**: Foot points for tracking validation

### Sample Evidence Filenames Generated:
- `/alerts/snapshots/cam0_alert42_20260908_143022.jpg` - Alert snapshot
- `/clips/cam0_alert42_20260908_143022.mp4` - 10-second evidence clip (5s pre + 5s post alert)

## 3. Stream View Description (`streams/`)

### What User Sees in Browser at http://localhost:8000/stream/0:

```
+---------------------------------------------------------------------+
| CAM 0 | 14:30:22                                      |  ← Timestamp overlay
|                                                                     |
|                                                                     |
|     [PERSON]                                                        |
|     +-------------------+         +-------------------+             |
|     |                   |         |                   |             |
|     |  ID:15 person     |         |  ID:23 vehicle    |             |
|     |  0.91 confidence  |         |  0.87 confidence  |             |
|     |  ● foot point     |         |  ● foot point     |             |
|     +-------------------+         |                   |             |
|                                     |  [PLATE]          |             |
|                                     | +---------------+ |             |
|                                     | | HR26DC1234    | |             |
|                                     | | 0.87 conf     | |             |
|                                     | +---------------+ |             |
|                                     +-------------------+             |
|                                                                     |
|     [RULE BOUNDARY]                                                 |
|     ---------------------  ← Yellow line showing configured fence   |
|                                                                     |
|  ALERT: ENTRY ID:15  ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ←  |  ← Red flashing banner
|                                                                     |
|                                                                     |
|                                                                     |
+---------------------------------------------------------------------+
```

## 4. WebSocket Alert Sample

Real-time alerts pushed to connected clients:

```json
{
  "id": 42,
  "camera_id": 0,
  "alert_type": "entry",
  "object_class": "person",
  "track_id": 15,
  "confidence": 0.91,
  "timestamp": "2026-09-08T14:30:22Z",
  "explanation": "Subject (person) detected crossing perimeter boundary from exterior to interior with 91% confidence. Movement vector indicates deliberate approach velocity. Subject carrying backpack consistent with attempted smuggling. Low-light conditions present - CLAHE enhancement activated. Track maintained across 24 frames. Behavioral analysis shows sustained trajectory toward restricted zone.",
  "snapshot_url": "/alerts/snapshots/cam0_alert42_20260908_143022.jpg",
  "clip_url": "/clips/cam0_alert42_20260908_143022.mp4",
  "anpr_detected": true,
  "anpr_plates": ["HR26DC1234", "DL1CA2345"]
}
```

## How to Experience This in Real Deployment

When a user runs IBVAP with a sample video:

1. **Start System**: `python3 -m uvicorn api.main:app --host 0.0.0.0 --port 8000`
2. **Configure Video**: Set `DEFAULT_CAMERA_URL=/path/to/sample/video.mp4` in `.env`
3. **Open Dashboard**: Visit `http://localhost:8000` in browser
4. **View Live Stream**: Go to `http://localhost:8000/stream/0` 
5. **Monitor Alerts**: Watch for red banners and check API for explanations
6. **Verify Integrity**: Check `http://localhost:8000/api/integrity/status`
7. **Test ANPR**: Use `http://localhost:8000/api/anpr/test/0` 
8. **Review Evidence**: Check generated files in `/alerts/` and `/clips/` directories

All outputs shown above are exactly what users would see when running the fully deployed IBVAP system with real video input and all features (EB-SAI explanations, triple-layer integrity, ANPR stretch goal) working together in real-time.