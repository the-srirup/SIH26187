/**
 * IBVAP Dashboard — JavaScript
 * Handles live camera tiles, WebSocket alerts, fence drawing, and integrity verification.
 */

const API_BASE = '';
const WS_URL = `ws://${location.host}/ws/alerts`;
const STREAM_FPS = 5;

// ---- WebSocket ----
let ws = null;
let reconnectAttempts = 0;
const MAX_RECONNECT = 5;

function connectWS() {
    ws = new WebSocket(WS_URL);

    ws.onopen = () => {
        reconnectAttempts = 0;
        console.log('WebSocket connected');
    };

    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.type === 'alert') {
            prependAlert(msg.data);
            showToast(`🚨 ${msg.data.alert_type.toUpperCase()} — ${msg.data.object_class || 'Unknown'}`, 'alert');
        }
    };

    ws.onclose = () => {
        if (reconnectAttempts < MAX_RECONNECT) {
            reconnectAttempts++;
            setTimeout(connectWS, 2000 * reconnectAttempts);
        }
    };

    ws.onerror = () => ws.close();
}

// ---- Clock ----
function updateClock() {
    const now = new Date();
    document.getElementById('clock').textContent = now.toLocaleTimeString('en-US', { hour12: false }) + ' UTC';
}
setInterval(updateClock, 1000);
updateClock();

// ---- Camera Grid ----
async function loadCameras() {
    const grid = document.getElementById('camera-grid');
    try {
        const res = await fetch(`${API_BASE}/api/cameras`);
        const cameras = await res.json();

        if (cameras.length === 0) {
            grid.innerHTML = '<div class="camera-tile loading"><div class="spinner"></div><p>No cameras configured. Add one below!</p></div>';
            return;
        }

        grid.innerHTML = cameras.map(cam => `
            <div class="camera-tile">
                <img src="/stream/${cam.id}" alt="${cam.name}" loading="lazy">
                <div class="cam-label">
                    <span class="cam-name">${cam.name}</span>
                    <span class="cam-status">${cam.is_active ? '● LIVE' : '○ OFF'}</span>
                </div>
            </div>
        `).join('');
    } catch (err) {
        grid.innerHTML = '<div class="camera-tile loading"><p>Error loading cameras</p></div>';
        console.error(err);
    }
}

// ---- Alert Feed ----
let alertCount = 0;

async function loadAlerts() {
    const feed = document.getElementById('alert-feed');
    const filter = document.getElementById('alert-filter').value;
    try {
        let url = `${API_BASE}/api/alerts?limit=50`;
        if (filter) url += `&alert_type=${filter}`;
        const res = await fetch(url);
        const alerts = await res.json();
        renderAlerts(alerts);
    } catch (err) {
        console.error(err);
    }
}

function renderAlerts(alerts) {
    const feed = document.getElementById('alert-feed');
    if (alerts.length === 0) {
        feed.innerHTML = '<p class="empty-state">No alerts yet</p>';
        return;
    }
    feed.innerHTML = alerts.map(a => createAlertCard(a)).join('');
}

function prependAlert(alert) {
    alertCount++;
    const feed = document.getElementById('alert-feed');
    if (feed.querySelector('.empty-state')) {
        feed.innerHTML = '';
    }
    const card = document.createElement('div');
    card.className = `alert-card alert-${alert.alert_type}`;
    card.onclick = () => openAlertModal(alert);
    card.innerHTML = `
        <div class="alert-header">
            <span class="alert-type">${alert.alert_type.replace('_', ' ').toUpperCase()}</span>
            <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
        </div>
        <div class="alert-details">
            ${alert.object_class || 'N/A'} | Track #${alert.track_id || '?'} | ${alert.confidence ? (alert.confidence * 100).toFixed(0) + '%' : ''}
        </div>
    `;
    feed.prepend(card);
}

function createAlertCard(alert) {
    return `
        <div class="alert-card alert-${alert.alert_type}" onclick="openAlertModal(${JSON.stringify(alert).replace(/"/g, '&quot;')})">
            <div class="alert-header">
                <span class="alert-type">${alert.alert_type.replace('_', ' ').toUpperCase()}</span>
                <span class="alert-time">${new Date(alert.timestamp).toLocaleTimeString()}</span>
            </div>
            <div class="alert-details">
                ${alert.object_class || 'N/A'} | Track #${alert.track_id || '?'} | Confidence: ${alert.confidence ? (alert.confidence * 100).toFixed(1) + '%' : 'N/A'}
            </div>
        </div>
    `;
}

// ---- Alert Detail Modal ----
function openAlertModal(alert) {
    const modal = document.getElementById('alert-modal');
    const body = document.getElementById('modal-body');

    body.innerHTML = `
        <h2>Alert #${alert.id}</h2>
        <dl class="modal-metadata">
            <dt>Type</dt><dd>${alert.alert_type.replace('_', ' ').toUpperCase()}</dd>
            <dt>Camera</dt><dd>${alert.camera_name}</dd>
            <dt>Object</dt><dd>${alert.object_class || 'N/A'}</dd>
            <dt>Track ID</dt><td>${alert.track_id || 'N/A'}</dd>
            <dt>Confidence</dt><dd>${alert.confidence ? (alert.confidence * 100).toFixed(1) + '%' : 'N/A'}</dd>
            <dt>Timestamp</dt><dd>${new Date(alert.timestamp).toLocaleString()}</dd>
            <dt>Hash</dt><dd><code style="font-size:0.7rem;word-break:break-all">${alert.hash}</code></dd>
        </dl>
        ${alert.snapshot_path ? `<img src="${alert.snapshot_path}" class="modal-video" alt="Snapshot">` : ''}
        ${alert.clip_path ? `<video src="${alert.clip_path}" class="modal-video" controls></video>` : ''}
        <div style="margin-top:16px;display:flex;gap:8px">
            <a href="${alert.clip_path}" class="btn btn-primary" ${!alert.clip_path ? 'style="display:none"' : ''}>Download Clip</a>
            <a href="${alert.snapshot_path}" class="btn btn-outline" ${!alert.snapshot_path ? 'style="display:none"' : ''}>Download Snapshot</a>
        </div>
    `;
    modal.classList.add('show');
}

function closeModal() {
    document.getElementById('alert-modal').classList.remove('show');
}

// ---- Add Camera ----
function openAddCamera() {
    document.getElementById('camera-modal').classList.add('show');
    document.getElementById('cam-name').focus();
}

function closeCameraModal() {
    document.getElementById('camera-modal').classList.remove('show');
}

async function submitCamera(e) {
    e.preventDefault();
    const name = document.getElementById('cam-name').value;
    const url = document.getElementById('cam-url').value;
    const location = document.getElementById('cam-location').value;

    try {
        const formData = new FormData();
        formData.append('name', name);
        formData.append('url', url);
        formData.append('location', location);

        const res = await fetch(`${API_BASE}/api/cameras`, {
            method: 'POST',
            body: formData,
        });
        if (res.ok) {
            closeCameraModal();
            showToast('✅ Camera added successfully', 'success');
            loadCameras();
            document.getElementById('camera-form').reset();
        } else {
            showToast('❌ Failed to add camera', 'error');
        }
    } catch (err) {
        showToast('❌ Connection error', 'error');
    }
}

// ---- Fence / Zone Drawing Canvas ----
let drawMode = 'line';
let drawPoints = [];
let drawCtx = null;

function setDrawMode(mode) {
    drawMode = mode;
    document.getElementById('btn-draw-line').classList.toggle('active', mode === 'line');
    document.getElementById('btn-draw-zone').classList.toggle('active', mode === 'zone');
    document.getElementById('draw-hint').textContent = mode === 'line'
        ? 'Click two points to define a line'
        : 'Click to place polygon points — double-click to finish';
    clearDrawing();
}

function openCanvasModal(cameraId) {
    document.getElementById('canvas-modal').classList.add('show');
    initDrawCanvas(cameraId);
}

function closeCanvasModal() {
    document.getElementById('canvas-modal').classList.remove('show');
    drawPoints = [];
}

function initDrawCanvas(cameraId) {
    const canvas = document.getElementById('draw-canvas');
    drawCtx = canvas.getContext('2d');
    canvas.width = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
    drawPoints = [];
    drawCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Load current frame as background
    const img = new Image();
    img.onload = () => {
        drawCtx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = `/stream/${cameraId}`;
}

const drawCanvas = document.getElementById('draw-canvas');
drawCanvas.addEventListener('click', (e) => {
    const rect = drawCanvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) * (drawCanvas.width / rect.width);
    const y = (e.clientY - rect.top) * (drawCanvas.height / rect.height);

    drawPoints.push([Math.round(x), Math.round(y)]);
    redrawCanvas();
    updateCoordsDisplay();

    if (drawMode === 'line' && drawPoints.length >= 2) {
        showToast('Line defined — click Save to confirm', 'info');
    }
});

drawCanvas.addEventListener('dblclick', () => {
    if (drawMode === 'zone' && drawPoints.length >= 3) {
        showToast('Zone complete — click Save to confirm', 'info');
    }
});

function redrawCanvas() {
    if (!drawCtx || drawPoints.length === 0) return;
    // Re-initialize is better, but for now just draw points and lines
    const canvas = document.getElementById('draw-canvas');
    drawCtx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw points
    drawPoints.forEach((pt, i) => {
        drawCtx.beginPath();
        drawCtx.arc(pt[0], pt[1], 5, 0, Math.PI * 2);
        drawCtx.fillStyle = drawMode === 'line' && i < 2 ? '#3b82f6' : '#f59e0b';
        drawCtx.fill();
        drawCtx.strokeStyle = 'white';
        drawCtx.lineWidth = 2;
        drawCtx.stroke();

        // Label point index
        drawCtx.fillStyle = 'white';
        drawCtx.font = '12px monospace';
        drawCtx.fillText(`P${i+1}`, pt[0] + 8, pt[1] - 8);
    });

    // Draw lines
    if (drawPoints.length >= 2) {
        drawCtx.beginPath();
        drawCtx.moveTo(drawPoints[0][0], drawPoints[0][1]);
        for (let i = 1; i < drawPoints.length; i++) {
            drawCtx.lineTo(drawPoints[i][0], drawPoints[i][1]);
        }
        drawCtx.strokeStyle = '#3b82f6';
        drawCtx.lineWidth = 3;
        drawCtx.stroke();
    }
}

function updateCoordsDisplay() {
    const display = document.getElementById('coords-display');
    if (drawPoints.length === 0) {
        display.textContent = '';
        return;
    }
    display.textContent = `Points (${drawPoints.length}): [${drawPoints.map(p => `[${p[0]}, ${p[1]}]`).join(', ')}]`;
}

function clearDrawing() {
    drawPoints = [];
    const canvas = document.getElementById('draw-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    updateCoordsDisplay();
}

async function saveDrawing() {
    if (drawPoints.length < 2) {
        showToast('Need at least 2 points for a line', 'error');
        return;
    }

    // Determine rule type
    const ruleType = drawMode === 'line' ? 'line' : 'zone';
    const cameraId = currentCameraId;
    let geometry;

    if (ruleType === 'line') {
        geometry = drawPoints.slice(0, 2);
    } else {
        geometry = drawPoints;
    }

    try {
        const res = await fetch(`${API_BASE}/api/cameras/${cameraId}/rules`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                rule_type: ruleType,
                geometry: geometry,
                params: {},
            }),
        });

        if (res.ok) {
            closeCanvasModal();
            showToast('✅ Rule saved successfully', 'success');
        } else {
            showToast('❌ Failed to save rule', 'error');
        }
    } catch (err) {
        showToast('❌ Connection error', 'error');
    }
}

let currentCameraId = 1;

// ---- Integrity Verification ----
async function verifyIntegrity() {
    const badge = document.getElementById('integrity-badge');
    const btn = document.getElementById('btn-integrity');

    btn.classList.add('verifying');
    btn.textContent = '⏳ Verifying...';

    try {
        const res = await fetch(`${API_BASE}/api/integrity/verify`);
        const data = await res.json();

        if (data.valid) {
            badge.className = 'badge badge-valid';
            badge.textContent = '✓ Verified';
            showToast('✅ Alert log integrity verified — chain intact', 'success');
        } else {
            badge.className = 'badge badge-invalid';
            badge.textContent = '✗ TAMPERED';
            showToast(`🚨 INTEGRITY BREACH at alert #${data.broken_at}!`, 'error');
        }
    } catch (err) {
        showToast('❌ Verification failed', 'error');
    }

    btn.classList.remove('verifying');
    btn.textContent = '🔗 Verify Log Integrity';
}

// ---- Toasts ----
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);

    setTimeout(() => {
        toast.style.opacity = '0';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

// ---- Init ----
document.addEventListener('DOMContentLoaded', () => {
    loadCameras();
    loadAlerts();
    connectWS();
    setInterval(loadCameras, 30000); // Refresh camera list every 30s
    setInterval(loadAlerts, 15000);   // Refresh alert list every 15s
});
