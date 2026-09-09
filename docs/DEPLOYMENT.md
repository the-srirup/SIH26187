# IBVAP Deployment Guide

## Production Deployment Options

### Option 1: Docker Container (Recommended for Production)

#### Single Container Deployment
```bash
# Build the production image
docker build -t ibvap:prod -f Dockerfile.prod .

# Run the container
docker run -d \
  --name ibvap \
  -p 8000:8000 \
  -v $(pwd)/alerts:/app/alerts \
  -v $(pwd)/clips:/app/clips \
  -v $(pwd)/snapshots:/app/snapshots \
  -v $(pwd)/videos:/app/videos \
  ibvap:prod
```

#### Docker Compose Deployment
```yaml
version: '3.8'

services:
  ibvap:
    build:
      context: .
      dockerfile: Dockerfile.prod
    ports:
      - "8000:8000"
    volumes:
      - ./alerts:/app/alerts
      - ./clips:/app/clips
      - ./snapshots:/app/snapshots
      - ./videos:/app/videos
    environment:
      - DATABASE_URL=sqlite:///./alerts.db
      - HOST=0.0.0.0
      - PORT=8000
      - TARGET_FPS=5
      - MODEL_PATH=yolo11n.pt
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Option 2: Direct Python Installation

#### Requirements
- Python 3.11+
- FFmpeg (for video processing)
- GPU drivers (optional, for accelerated inference)

#### Installation Steps
```bash
# Clone repository
git clone https://github.com/yourorg/ibvap.git
cd ibvap

# Install dependencies
pip install -r requirements.txt

# Download YOLO model (first run only)
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"

# Initialize database
python manage.py init

# Start the application
python manage.py run
```

### Option 3: Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ibvap
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ibvap
  template:
    metadata:
      labels:
        app: ibvap
    spec:
      containers:
      - name: ibvap
        image: ibvap:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: storage
          mountPath: /app/alerts
        - name: storage
          mountPath: /app/clips
        - name: storage
          mountPath: /app/snapshots
        - name: storage
          mountPath: /app/videos
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: ibvap-storage
---
apiVersion: v1
kind: Service
metadata:
  name: ibvap-service
spec:
  selector:
    app: ibvap
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

## Configuration

### Environment Variables
IBVAP uses environment variables for configuration. Create a `.env` file:

```env
# Core Settings
PROJECT_NAME=IBVAP
VERSION=1.0.0

# Database
DATABASE_URL=sqlite:///./alerts.db

# Storage Paths
BASE_DIR=/app
STATIC_DIR=/app/static
DASHBOARD_DIR=/app/dashboard
ALERTS_DIR=/app/alerts
CLIPS_DIR=/app/clips
SNAPSHOTS_DIR=/app/alerts/snapshots
VIDEOS_DIR=/app/videos

# Model Settings
MODEL_PATH=yolo11n.pt
TRACKER=bytetrack.yaml

# Video Processing
BUFFER_SIZE=150
POST_ALERT_FRAMES=50
FRAME_WIDTH=640
FRAME_HEIGHT=480
TARGET_FPS=5

# Detection Settings
MIN_OBJECT_AREA=300
DEFAULT_CONFIDENCE=0.25

# Rule Settings
DEBOUNCE_SECONDS=10.0
LOITER_SECONDS=60

# Server Settings
HOST=0.0.0.0
PORT=8000
```

### Performance Tuning

#### For Low-Power Edge Devices
```env
TARGET_FPS=3              # Reduce to 3 FPS for lower power consumption
MODEL_PATH=yolo11n.pt     # Use nano model
BUFFER_SIZE=90            # Smaller buffer
POST_ALERT_FRAMES=30      # Fewer post-alert frames
```

#### For High-Performance Servers
```env
TARGET_FPS=15             # Higher FPS for smoother video
MODEL_PATH=yolo11l.pt     # Use larger model for better accuracy
BUFFER_SIZE=225           # Larger buffer
POST_ALERT_FRAMES=75      # More post-alert frames
MIN_OBJECT_AREA=100       # Detect smaller objects
```

## Hardware Requirements

### Minimum (Edge Deployment)
- CPU: Quad-core ARM Cortex-A53 or better
- RAM: 2GB
- Storage: 8GB eMMC or SD card
- Network: Ethernet or WiFi
- USB: For camera connectivity

### Recommended (Server Deployment)
- CPU: 6-core x86_64 or equivalent ARM
- RAM: 8GB+
- Storage: 64GB SSD
- GPU: NVIDIA Jetson or equivalent (for AI acceleration)
- Network: Gigabit Ethernet

### High-Performance (Data Center)
- CPU: 16-core Xeon or EPYC
- RAM: 32GB+
- Storage: 256GB NVMe SSD
- GPU: NVIDIA T4 or better
- Network: 10GbE

## Monitoring and Maintenance

### Health Checks
The application includes built-in health checks:
- `GET /health` - Basic service health
- `GET /api/system/info` - Detailed system information
- `GET /api/stats/advanced` - Performance metrics

### Log Management
Logs are written to stdout/stderr and can be collected by:
- Docker logging drivers
- Kubernetes logging stack
- External ELK/EFK stack

### Backup Procedures
1. Stop the IBVAP service
2. Backup the SQLite database (`alerts.db`)
3. Backup the alerts directory (contains evidence)
4. Restart the service

### Updates
To update IBVAP:
1. Pull latest code: `git pull`
2. Rebuild Docker image: `docker build -t ibvap:latest .`
3. Restart containers: `docker-compose up -d`

## Security Considerations

### Network Security
- Change default ports if needed
- Use firewall rules to restrict access
- Consider placing behind a reverse proxy (NGINX, Traefik)
- Enable HTTPS using Let's Encrypt or corporate certificates

### Data Security
- The hash chain provides tamper-evident logging
- Consider encrypting the alerts directory for additional security
- Regularly export and store chain hashes off-site for legal purposes

### API Security
- All endpoints require proper authentication (to be implemented)
- Rate limiting prevents abuse
- Input validation prevents injection attacks

## Troubleshooting

### Common Issues

#### Camera Not Connecting
1. Verify camera URL is correct
2. Check network connectivity
3. Ensure camera supports RTSP/HTTP streaming
4. Test with `ffmpeg -i "camera_url" -t 5 test.mp4`

#### High CPU Usage
1. Reduce TARGET_FPS in configuration
2. Use smaller YOLO model (yolo11n.pt vs yolo11l.pt)
3. Increase DEBOUNCE_SECONDS to reduce processing
4. Ensure hardware acceleration is enabled if available

#### Low Detection Accuracy
1. Ensure adequate lighting; enable low-light enhancement
2. Check camera focus and positioning
3. Verify minimum object area is appropriate for use case
4. Consider fine-tuning model on domain-specific data

#### Storage Issues
1. Monitor disk space usage
2. Implement log rotation for old evidence
3. Consider compressing old video clips
4. Use external storage for long-term archival

## Performance Benchmarks

### Expected Performance
| Configuration | FPS | Latency | Power Consumption |
|---------------|-----|---------|-------------------|
| Raspberry Pi 4 | 3-5 | 100-200ms | 5-7W |
| Jetson Nano | 10-15 | 50-100ms | 5-10W |
| Jetson Xavier | 20-30 | 30-60ms | 10-15W |
| Desktop CPU | 15-25 | 40-80ms | 30-60W |
| GPU Server | 30-60 | 20-40ms | 100-200W |

### Scalability
- Single instance: 4-6 cameras (depending on hardware)
- Multi-instance: Horizontal scaling with load balancer
- Cloud deployment: Auto-scaling groups based on alert volume

## Support and Maintenance

### Regular Maintenance Tasks
- Weekly: Check disk space and cleanup old evidence
- Monthly: Review logs for patterns and anomalies
- Quarterly: Test backup and restore procedures
- Annually: Hardware inspection and firmware updates

### Contact Information
For enterprise support, contact: security@yourorg.com
For community support: https://github.com/yourorg/ibvap/discussions

## License
IBVAP is released under the MIT License. See LICENSE file for details.