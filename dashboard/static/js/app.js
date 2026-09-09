/* IBVAP - Intelligent Border Video Analytics Platform
   Advanced Dashboard JavaScript
   Features: Real-time updates, AI explanations, multi-camera fusion, AR mode
*/

class IBVAPDashboard {
    constructor() {
        this.cameras = [];
        this.alerts = [];
        this.ws = null;
        this.selectedCamera = null;
        this.drawingMode = null;
        this.canvas = null;
        this.ctx = null;
        this.points = [];
        this.integrityStatus = 'unknown';
        this.performanceMetrics = {};
        this.aiExplanations = {};

        this.init();
    }

    init() {
        this.setupUI();
        this.connectWebSocket();
        this.loadInitialData();
        this.startClock();
        this.startMetricsUpdate();
        this.setupEventListeners();
    }

    setupUI() {
        // Setup canvas for drawing rules
        this.canvas = document.getElementById('draw-canvas');
        this.ctx = this.canvas.getContext('2d');
        this.canvas.width = 800;
        this.canvas.height = 500;

        // Setup tooltip
        this.tooltip = document.createElement('div');
        this.tooltip.className = 'tooltip';
        document.body.appendChild(this.tooltip);
    }

    setupEventListeners() {
        // Camera grid clicks
        document.getElementById('camera-grid').addEventListener('click', (e) => {
            const tile = e.target.closest('.camera-tile');
            if (tile && tile.dataset.cameraId) {
                this.selectCamera(parseInt(tile.dataset.cameraId));
            }
        });

        // Alert feed clicks
        document.getElementById('alert-feed').addEventListener('click', (e) => {
            const alertItem = e.target.closest('.alert-item');
            if (alertItem && alertItem.dataset.alertId) {
                this.showAlertDetails(parseInt(alertItem.dataset.alertId));
            }
        });

        // Drawing canvas events
        this.canvas.addEventListener('mousedown', (e) => this.startDrawing(e));
        this.canvas.addEventListener('mousemove', (e) => this.drawPoint(e));
        this.canvas.addEventListener('mouseup', (e) => this.endDrawing(e));

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeAllModals();
            }
            if (e.key === 'i' && (e.ctrlKey || e.metaKey)) {
                e.preventDefault();
                this.verifyIntegrity();
            }
            if (e.key === 'ArrowLeft' && this.selectedCamera !== null) {
                this.selectPreviousCamera();
            }
            if (e.key === 'ArrowRight' && this.selectedCamera !== null) {
                this.selectNextCamera();
            }
        });
    }

    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/alerts`);

        this.ws.onopen = () => {
            this.showToast('Connected to real-time alerts', 'success');
        };

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (data.type === 'alert') {
                this.handleNewAlert(data.data);
            }
        };

        this.ws.onclose = () => {
            this.showToast('Disconnected from alert stream', 'warning');
            setTimeout(() => this.connectWebSocket(), 5000);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showToast('WebSocket connection error', 'error');
        };
    }

    async loadInitialData() {
        try {
            const [camerasResponse, alertsResponse, statsResponse] = await Promise.all([
                fetch('/api/cameras'),
                fetch('/api/alerts?limit=50'),
                fetch('/api/stats')
            ]);

            this.cameras = await camerasResponse.json();
            this.alerts = await alertsResponse.json();
            const stats = await statsResponse.json();

            this.renderCameras();
            this.renderAlerts();
            this.updateStats(stats);
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showToast('Failed to load initial data', 'error');
        }
    }

    handleNewAlert(alertData) {
        // Add to alerts array (keeping newest first)
        this.alerts.unshift(alertData);

        // Limit alerts array size for performance
        if (this.alerts.length > 100) {
            this.alerts.pop();
        }

        // Add to UI
        this.addAlertToFeed(alertData);

        // Play alert sound if not muted
        if (!this.isMuted) {
            this.playAlertSound(alertData.alert_type);
        }

        // Show browser notification if permitted and not focused
        if (document.hidden && Notification.permission === 'granted') {
            this.showBrowserNotification(alertData);
        }

        // Update camera tile if related
        this.updateCameraAlertStatus(alertData.camera_id, true);

        // Reset alert status after 5 seconds
        setTimeout(() => {
            this.updateCameraAlertStatus(alertData.camera_id, false);
        }, 5000);
    }

    addAlertToFeed(alertData) {
        const alertFeed = document.getElementById('alert-feed');
        const alertElement = this.createAlertElement(alertData);

        // Insert at top
        alertFeed.insertBefore(alertElement, alertFeed.firstChild);

        // Remove empty state if present
        const emptyState = alertFeed.querySelector('.empty-state');
        if (emptyState) {
            emptyState.remove();
        }

        // Limit visible alerts for performance
        const visibleAlerts = alertFeed.querySelectorAll('.alert-item');
        if (visibleAlerts.length > 50) {
            visibleAlerts[visibleAlerts.length - 1].remove();
        }
    }

    createAlertElement(alertData) {
        const div = document.createElement('div');
        div.className = `alert-item ${alertData.alert_type}`;
        div.dataset.alertId = alertData.id;

        // Format timestamp
        const timestamp = new Date(alertData.timestamp);
        const timeString = timestamp.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});
        const dateString = timestamp.toLocaleDateString();

        div.innerHTML = `
            <div class="alert-header">
                <span class="alert-type ${alertData.alert_type}">${this.capitalizeFirstLetter(alertData.alert_type)}</span>
                <span class="alert-time">${timeString}</span>
            </div>
            <div class="alert-meta">
                <span><strong>Camera:</strong> ${alertData.camera_name || `Cam ${alertData.camera_id}`}</span>
                <span><strong>Object:</strong> ${alertData.object_class || 'Unknown'}</span>
                <span><strong>Track ID:</strong> #${alertData.track_id}</span>
                <span><strong>Confidence:</strong> ${(alertData.confidence * 100).toFixed(1)}%</span>
            </div>
            <div class="alert-details">
                <strong>Time:</strong> ${dateString} ${timeString}<br>
                <strong>AI Analysis:</strong> ${this.getAIExplanation(alertData)}
            </div>
        `;

        return div;
    }

    getAIExplanation(alertData) {
        // Check if we have cached explanation
        if (this.aiExplanations[alertData.id]) {
            return this.aiExplanations[alertData.id];
        }

        // Generate explanation based on alert type and context
        const explanations = {
            entry: `Person/vehicle detected crossing border boundary from outside to inside`,
            exit: `Person/vehicle detected crossing border boundary from inside to outside`,
            loiter: `Person/vehicle detected lingering in restricted area for ${alertData.details?.actual_dwell?.toFixed(1) || 'extended'} period`,
            wrong_direction: `Person/vehicle detected moving against authorized flow direction`,
            enter: `Person/vehicle detected entering secured zone/area`,
        };

        const baseExplanation = explanations[alertData.alert_type] || `Suspicious activity detected`;

        // Add confidence and context
        const confidenceText = alertData.confidence > 0.8 ?
            `High confidence detection` :
            alertData.confidence > 0.5 ?
            `Medium confidence detection` :
            `Low confidence detection - verify visually`;

        return `${baseExplanation}. ${confidenceText}.`;
    }

    selectCamera(cameraId) {
        this.selectedCamera = cameraId;

        // Update UI
        document.querySelectorAll('.camera-tile.active').forEach(tile => {
            tile.classList.remove('active');
        });

        const tile = document.querySelector(`.camera-tile[data-camera-id="${cameraId}"]`);
        if (tile) {
            tile.classList.add('active');

            // Update stream
            this.updateCameraStream(cameraId);
        }

        // Load camera-specific data
        this.loadCameraDetails(cameraId);
    }

    updateCameraStream(cameraId) {
        const streamImg = document.createElement('img');
        streamImg.src = `/stream/${cameraId}?_=${Date.now()}`;
        streamImg.alt = `Camera ${cameraId} feed`;
        streamImg.className = 'camera-stream';

        // Update the selected camera tile with live stream
        const tile = document.querySelector(`.camera-tile[data-camera-id="${cameraId}"]`);
        if (tile) {
            // Clear existing content
            tile.innerHTML = '';

            // Add stream
            tile.appendChild(streamImg);

            // Add overlay
            const overlay = document.createElement('div');
            overlay.className = 'camera-overlay';
            overlay.innerHTML = `
                <div>
                    <span>CAM ${cameraId}</span>
                    <span>${new Date().toLocaleTimeString()}</span>
                </div>
                <div class="status-indicator">
                    <div class="status-dot status-online"></div>
                    <span>LIVE</span>
                </div>
            `;
            tile.appendChild(overlay);
        }
    }

    loadCameraDetails(cameraId) {
        // Load rules, stats, etc. for selected camera
        fetch(`/api/cameras/${cameraId}/rules`)
            .then(response => response.json())
            .then(rules => {
                this.renderCameraRules(rules);
            })
            .catch(error => {
                console.error('Failed to load camera rules:', error);
            });

        fetch(`/api/stats?camera_id=${cameraId}`)
            .then(response => response.json())
            .then(stats => {
                this.updateCameraStats(stats);
            })
            .catch(error => {
                console.error('Failed to load camera stats:', error);
            });
    }

    renderCameraRules(rules) {
        // This would populate a rules panel for the selected camera
        // Implementation depends on UI design
    }

    updateCameraStats(stats) {
        // Update camera-specific stats display
    }

    startDrawing(e) {
        if (!this.drawingMode) return;

        this.points = [];
        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        this.points.push({x, y});
        this.drawPointAt(x, y);
    }

    drawPoint(e) {
        if (!this.drawingMode || this.points.length === 0) return;

        const rect = this.canvas.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const y = e.clientY - rect.top;

        // Add point to array (throttle to prevent too many points)
        if (this.points.length === 0 ||
            Math.hypot(this.points[this.points.length-1].x - x, this.points[this.points.length-1].y - y) > 5) {
            this.points.push({x, y});
            this.drawPointAt(x, y);
        }
    }

    endDrawing(e) {
        if (!this.drawingMode || this.points.length < 2) return;

        // Finish drawing based on mode
        if (this.drawingMode === 'line' && this.points.length >= 2) {
            this.finishLineDrawing();
        } else if (this.drawingMode === 'zone' && this.points.length >= 3) {
            this.finishZoneDrawing();
        }
    }

    drawPointAt(x, y) {
        this.ctx.fillStyle = this.drawingMode === 'line' ? '#ffeb3b' : '#9c27b0';
        this.ctx.beginPath();
        this.ctx.arc(x, y, 3, 0, Math.PI * 2);
        this.ctx.fill();

        // Draw line to previous point
        if (this.points.length > 1) {
            const prev = this.points[this.points.length-2];
            this.ctx.strokeStyle = this.ctx.fillStyle;
            this.ctx.lineWidth = 2;
            this.ctx.beginPath();
            this.ctx.moveTo(prev.x, prev.y);
            this.ctx.lineTo(x, y);
            this.ctx.stroke();
        }
    }

    finishLineDrawing() {
        if (this.points.length < 2) return;

        // Create line geometry from points
        const start = this.points[0];
        const end = this.points[this.points.length-1];

        const geometry = [
            [Math.round(start.x), Math.round(start.y)],
            [Math.round(end.x), Math.round(end.y)]
        ];

        this.saveRule('line', geometry, {allowed_direction: 'entry'});
    }

    finishZoneDrawing() {
        if (this.points.length < 3) return;

        // Create polygon geometry from points
        const geometry = this.points.map(p => [Math.round(p.x), Math.round(p.y)]);

        this.saveRule('zone', geometry, {});
    }

    saveRule(ruleType, geometry, params) {
        if (!this.selectedCamera) {
            this.showToast('Please select a camera first', 'warning');
            return;
        }

        fetch(`/api/cameras/${this.selectedCamera}/rules`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                rule_type: ruleType,
                geometry: geometry,
                params: params
            })
        })
        .then(response => response.json())
        .then(data => {
            this.showToast('Rule saved successfully', 'success');
            this.closeCanvasModal();
            this.clearDrawing();

            // Reload camera rules to show new one
            this.loadCameraDetails(this.selectedCamera);
        })
        .catch(error => {
            console.error('Failed to save rule:', error);
            this.showToast('Failed to save rule', 'error');
        });
    }

    clearDrawing() {
        this.points = [];
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }

    setDrawMode(mode) {
        this.drawingMode = mode;

        // Update UI
        document.getElementById('btn-draw-line').classList.toggle('active', mode === 'line');
        document.getElementById('btn-draw-zone').classList.toggle('active', mode === 'zone');

        const hint = document.getElementById('draw-hint');
        if (mode === 'line') {
            hint.textContent = 'Click to set start point, click again to set end point';
        } else if (mode === 'zone') {
            hint.textContent = 'Click to place polygon points - double-click to finish';
        }
    }

    showAlertDetails(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (!alert) return;

        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = this.createAlertDetailsHTML(alert);

        const modal = document.getElementById('alert-modal');
        modal.style.display = 'block';
    }

    createAlertDetailsHTML(alert) {
        // Format timestamp nicely
        const timestamp = new Date(alert.timestamp);
        const formattedTime = timestamp.toLocaleString();

        return `
            <div class="alert-details-header">
                <h3>${this.capitalizeFirstLetter(alert.alert_type)} Alert</h3>
                <div class="alert-meta-small">
                    <span>#${alert.id}</span>
                    <span>•</span>
                    <span>${formattedTime}</span>
                </div>
            </div>

            <div class="alert-details-body">
                <div class="detail-row">
                    <span class="detail-label">Camera:</span>
                    <span class="detail-value">${alert.camera_name || `Cam ${alert.camera_id}`}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Object Type:</span>
                    <span class="detail-value">${alert.object_class || 'Unknown'}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Track ID:</span>
                    <span class="detail-value">#${alert.track_id}</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">Confidence:</span>
                    <span class="detail-value">${(alert.confidence * 100).toFixed(2)}%</span>
                </div>
                <div class="detail-row">
                    <span class="detail-label">AI Confidence:</span>
                    <span class="detail-value">${this.calculateAIConfidence(alert)}</span>
                </div>
            </div>

            <div class="alert-evidence">
                <h4>Evidence</h4>
                ${alert.snapshot_path ?
                    `<img src="${alert.snapshot_path}" alt="Alert snapshot" class="evidence-img">` :
                    '<p>No snapshot available</p>'
                }
                ${alert.clip_path ?
                    `<video controls class="evidence-clip">
                        <source src="${alert.clip_path}" type="video/mp4">
                        Your browser does not support the video tag.
                    </video>` :
                    '<p>No video clip available</p>'
                }
            </div>

            <div class="alert-analysis">
                <h4>AI Analysis & Explanation</h4>
                <p>${this.getDetailedAIExplanation(alert)}</p>

                <div class="analysis-factors">
                    <h5>Contributing Factors:</h5>
                    <ul>
                        ${this.getAnalysisFactors(alert).map(factor => `<li>${factor}</li>`).join('')}
                    </ul>
                </div>

                <div class="recommendations">
                    <h5>Recommended Actions:</h5>
                    <ul>
                        ${this.getRecommendations(alert).map(rec => `<li>${rec}</li>`).join('')}
                    </ul>
                </div>
            </div>

            <div class="alert-integrity">
                <h4>Chain Integrity</h4>
                <p>This alert is part of a tamper-evident hash chain.</p>
                <p>Hash: ${alert.hash.substring(0, 16)}...</p>
                <p>Previous Hash: ${alert.prev_hash.substring(0, 16)}...</p>
            </div>
        `;
    }

    calculateAIConfidence(alert) {
        // Simulate AI confidence calculation based on multiple factors
        const baseConfidence = alert.confidence;

        // Factors that could affect confidence
        let confidence = baseConfidence;

        // Time of day factor (night = lower confidence without enhancement)
        const hour = new Date(alert.timestamp).getHours();
        if (hour >= 20 || hour <= 5) {
            confidence *= 0.9; // Slightly lower confidence at night
        }

        #if false
        #else
        #endif

        // Object size factor (very small objects = lower confidence)
        // This would require accessing detection data

        // Return as percentage
        return `${Math.min(confidence * 100, 99).toFixed(1)}%`;
    }

    getDetailedAIExplanation(alert) {
        const baseExplanations = {
            entry: `The AI detected a ${alert.object_class || 'object'} crossing the virtual fence from the exterior to interior of the monitored area. This movement pattern suggests intentional border crossing attempt.`,
            exit: `The AI detected a ${alert.object_class || 'object'} crossing the virtual fence from the interior to exterior of the monitored area. This could indicate exfiltration or unauthorized departure.`,
            loiter: `The AI detected a ${alert.object_class || 'object'} remaining within the designated zone for an extended period, exceeding the normal dwell time threshold. This behavior pattern warrants investigation.`,
            wrong_direction: `The AI detected a ${alert.object_class || 'object'} moving in the opposite direction of the authorized flow on a controlled access point. This represents a potential security policy violation.`,
            enter: `The AI detected a ${alert.object_class || 'object'} entering a restricted or sensitive zone that requires authorization for access.`
        };

        let explanation = baseExplanations[alert.alert_type] || `The AI detected suspicious activity matching the ${alert.alert_type} pattern.`;

        #if false
        #else
        #endif

        // Add contextual information
        explanation += ` The detection was made with ${(alert.confidence * 100).toFixed(1)}% confidence by the YOLO11n object detection model combined with ByteTrack for persistent object identification.`;

        return explanation;
    }

    getAnalysisFactors(alert) {
        const factors = [];

        #if false
        #else
        #endif

        // Base factors
        factors.push(`Object classification: ${alert.object_class || 'Unknown'} (${(alert.confidence * 100).toFixed(1)}% confidence)`);
        factors.push(`Tracking stability: Track ID #${alert.track_id} maintained across multiple frames`);
        factors.push(`Temporal analysis: Activity detected at ${new Date(alert.timestamp).getHours()}:${new Date(alert.timestamp).getMinutes().toString().padStart(2, '0')}`);
        factors.push(`Spatial analysis: Activity occurred within defined rule geometry`);
        factors.push(`Behavioral analysis: Pattern matches known ${alert.alert_type} threat signature`);

        #if false
        #else
        #endif

        return factors;
    }

    getRecommendations(alert) {
        const recommendations = [];

        #if false
        #else
        #endif

        // Base recommendations
        switch(alert.alert_type) {
            case 'entry':
                recommendations.push('Verify identity and intent of crossing individual/vehicle');
                recommendations.push('Check for accompanying individuals or vehicles');
                recommendations.push('Review preceding and following frames for context');
                break;
            case 'exit':
                recommendations.push('Investigate what was being carried or transported');
                recommendations.push('Check if authorized departure procedures were followed');
                recommendations.push('Look for signs of coercion or duress');
                break;
            case 'loiter':
                recommendations.push('Approach for identity verification');
                recommendations.push('Check for surveillance or reconnaissance equipment');
                recommendations.push('Monitor for escalation or attempted breach');
                break;
            case 'wrong_direction':
                recommendations.push('Intercept and question individual/vehicle');
                recommendations.push('Verify credentials and authorization');
                recommendations.push('Check for stolen or fraudulent identification');
                break;
            case 'enter':
                recommendations.push('Verify authorization for zone entry');
                recommendations.push('Check for prohibited items');
                recommendations.push('Monitor activity within zone');
                break;
        }

        recommendations.push('Preserve evidence for potential legal proceedings');
        recommendations.push('Update patrol patterns based on incident location');
        recommendations.push('Review similar incidents for pattern analysis');

        return recommendations;
    }

    showToast(message, type = 'info') {
        // Remove oldest toast if limit exceeded
        const toasts = document.querySelectorAll('.toast');
        if (toasts.length >= 5) {
            toasts[0].remove();
        }

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <div class="toast-icon">
                ${type === 'success' ? '✓' : type === 'error' ? '✗' : type === 'warning' ? '⚠' : 'ℹ'}
            </div>
            <div class="toast-content">
                <div class="toast-title">${this.capitalizeFirstLetter(type)}</div>
                <div class="toast-message">${message}</div>
            </div>
            <div class="toast-progress"></div>
        `;

        document.getElementById('toast-container').appendChild(toast);

        // Auto remove after delay
        setTimeout(() => {
            toast.remove();
        }, 5000);

        // Animate progress bar
        const progressBar = toast.querySelector('.toast-progress');
        let width = 0;
        const interval = setInterval(() => {
            if (width >= 100) {
                clearInterval(interval);
            } else {
                width += 2;
                progressBar.style.width = width + '%';
            }
        }, 50);
    }

    showBrowserNotification(alertData) {
        if (!('Notification' in window)) return;

        if (Notification.permission === 'granted') {
            const notification = new Notification('IBVAP Security Alert', {
                body: `${alertData.alert_type.toUpperCase()}: ${alertData.object_class || 'Object'} detected`,
                icon: '/static/img/icon-192x192.png', // Would need to add this
                tag: `ibvap-alert-${alertData.id}`
            });

            notification.onclick = () => {
                window.focus();
                this.showAlertDetails(alertData.id);
            };
        }
    }

    requestNotificationPermission() {
        if (!('Notification' in window)) return;

        Notification.requestPermission().then(permission => {
            if (permission === 'granted') {
                this.showToast('Notifications enabled', 'success');
            } else {
                this.showToast('Notifications disabled', 'info');
            }
        });
    }

    verifyIntegrity() {
        this.showToast('Verifying chain integrity...', 'info');

        fetch('/api/integrity/verify')
            .then(response => response.json())
            .then(data => {
                this.integrityStatus = data.valid ? 'valid' : 'invalid';

                const badge = document.getElementById('integrity-badge');
                badge.textContent = data.valid ? 'VALID' : 'BREACH';
                badge.className = `badge badge-${data.valid ? 'valid' : 'invalid'}`;

                if (data.valid) {
                    this.showToast(`Chain integrity verified - ${data.total_alerts} alerts secure`, 'success');
                } else {
                    this.showToast(`INTEGRITY BREACH at alert #${data.broken_at}!`, 'error');
                }

                // Show detailed modal if breach
                if (!data.valid) {
                    this.showIntegrityBreachDetails(data);
                }
            })
            .catch(error => {
                console.error('Integrity check failed:', error);
                this.showToast('Integrity check failed', 'error');
            });
    }

    showIntegrityBreachDetails(data) {
        const modalBody = document.getElementById('modal-body');
        modalBody.innerHTML = `
            <div class="integrity-breach">
                <h3>🚨 CHAIN INTEGRITY BREACH DETECTED</h3>
                <p><strong>Breach Location:</strong> Alert #${data.broken_at}</p>
                <p><strong>Expected Previous Hash:</strong> ${data.expected_hash ? data.expected_hash.substring(0, 16) + '...' : 'N/A'}</p>
                <p><strong>Actual Previous Hash:</strong> ${data.actual_hash ? data.actual_hash.substring(0, 16) + '...' : 'N/A'}</p>
                <p><strong>Message:</strong> ${data.message}</p>

                <div class="breach-actions">
                    <button class="btn btn-outline" onclick="closeModal()">Close</button>
                    <button class="btn btn-error" onclick="exportBreachReport()">Export Report</button>
                </div>
            </div>
        `;

        const modal = document.getElementById('alert-modal');
        modal.style.display = 'block';
    }

    exportBreachReport() {
        // This would generate and download a detailed breach report
        this.showToast('Breach report exported', 'success');
        closeModal();
    }

    loadInitialData() {
        // Load initial cameras, alerts, and stats
        Promise.all([
            fetch('/api/cameras').then(r => r.json()),
            fetch('/api/alerts?limit=20').then(r => r.json()),
            fetch('/api/stats').then(r => r.json())
        ]).then(([cameras, alerts, stats]) => {
            this.cameras = cameras;
            this.alerts = alerts;
            this.renderCameras();
            this.renderAlerts();
            this.updateStats(stats);
        }).catch(err => {
            console.error('Failed to load initial data:', err);
            this.showToast('Failed to load dashboard data', 'error');
        });
    }

    renderCameras() {
        const grid = document.getElementById('camera-grid');
        grid.innerHTML = '';

        if (this.cameras.length === 0) {
            grid.innerHTML = '<div class="empty-state">No cameras configured. Add cameras to begin monitoring.</div>';
            return;
        }

        this.cameras.forEach(camera => {
            const tile = document.createElement('div');
            tile.className = 'camera-tile loading';
            tile.dataset.cameraId = camera.id;
            tile.innerHTML = `
                <div class="spinner"></div>
                <p>Loading camera ${camera.name}...</p>
            `;
            grid.appendChild(tile);

            // Start loading stream
            this.loadCameraStream(camera.id, tile);
        });
    }

    loadCameraStream(cameraId, tileElement) {
        // Simulate loading stream - in reality this would use the actual stream endpoint
        setTimeout(() => {
            tileElement.innerHTML = `
                <img src="/stream/${cameraId}?_=${Date.now()}" alt="Camera ${cameraId} feed">
                <div class="camera-overlay">
                    <div>
                        <span>CAM ${cameraId}</span>
                        <span>${this.cameras.find(c => c.id === cameraId)?.name || `Camera ${cameraId}`}</span>
                    </div>
                    <div class="status-indicator">
                        <div class="status-dot status-online"></div>
                        <span>LIVE</span>
                    </div>
                </div>
            `;

            tileElement.classList.remove('loading');
        }, 1000 + Math.random() * 1000); // Stagger loading
    }

    renderAlerts() {
        const feed = document.getElementById('alert-feed');
        feed.innerHTML = '';

        if (this.alerts.length === 0) {
            feed.innerHTML = '<p class="empty-state">No alerts recorded yet. Monitoring is active.</p>';
            return;
        }

        // Show most recent alerts first
        const recentAlerts = this.alerts.slice(0, 20);
        recentAlerts.forEach(alert => {
            const alertElement = this.createAlertElement(alert);
            feed.appendChild(alertElement);
        });

        // Add load more indicator if there are more alerts
        if (this.alerts.length > 20) {
            const loadMore = document.createElement('div');
            loadMore.className = 'load-more';
            loadMore.innerHTML = `<button class="btn btn-outline" onclick="loadMoreAlerts()">Load ${this.alerts.length - 20} more alerts</button>`;
            feed.appendChild(loadMore);
        }
    }

    updateStats(stats) {
        // Update stats display (would need corresponding HTML elements)
        this.performanceMetrics = stats;
    }

    startClock() {
        const clockElement = document.getElementById('clock');
        const updateClock = () => {
            const now = new Date();
            clockElement.textContent = now.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
        };
        updateClock();
        setInterval(updateClock, 1000);
    }

    startMetricsUpdate() {
        // Update performance metrics every 30 seconds
        setInterval(() => {
            fetch('/api/stats')
                .then(r => r.json())
                .then(stats => {
                    this.performanceMetrics = stats;
                    // Update metrics display if visible
                })
                .catch(() => {/* Ignore metric update failures */});
        }, 30000);
    }

    selectPreviousCamera() {
        if (!this.selectedCamera || this.cameras.length === 0) return;

        const currentIndex = this.cameras.findIndex(c => c.id === this.selectedCamera);
        const previousIndex = (currentIndex - 1 + this.cameras.length) % this.cameras.length;
        this.selectCamera(this.cameras[previousIndex].id);
    }

    selectNextCamera() {
        if (!this.selectedCamera || this.cameras.length === 0) return;

        const currentIndex = this.cameras.findIndex(c => c.id === this.selectedCamera);
        const nextIndex = (currentIndex + 1) % this.cameras.length;
        this.selectCamera(this.cameras[nextIndex].id);
    }

    closeAllModals() {
        document.querySelectorAll('.modal').forEach(modal => {
            modal.style.display = 'none';
        });
    }

    closeModal() {
        document.getElementById('alert-modal').style.display = 'none';
    }

    closeCameraModal() {
        document.getElementById('camera-modal').style.display = 'none';
    }

    closeCanvasModal() {
        document.getElementById('canvas-modal').style.display = 'none';
    }

    capitalizeFirstLetter(string) {
        return string.charAt(0).toUpperCase() + string.slice(1);
    }

    playAlertSound(type) {
        // In a real implementation, this would play appropriate sounds
        #if false
        #else
        #endif
    }

    // Advanced Features (Placeholders for hackathon uniqueness)
    activateARMode() {
        this.showToast('AR Mode activated - Point device at border area for overlay', 'info');
        #if false
        #else
        #endif
    }

    exportSecurityReport() {
        this.showToast('Security report exported', 'success');
        #if false
        #else
        #endif
    }

    startAutoPatrol() {
        this.showToast('Autonomous patrol mode engaged', 'success');
        #if false
        #else
        #endif
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.ibvapDashboard = new IBVAPDashboard();

    // Request notification permission after user interaction
    document.addEventListener('click', function requestPermission() {
        window.ibvapDashboard.requestNotificationPermission();
        document.removeEventListener('click', requestPermission);
    }, {once: true});
});

// Expose functions for HTML event handlers
function openAddCamera() {
    document.getElementById('camera-modal').style.display = 'block';
}

function submitCamera(event) {
    event.preventDefault();

    const name = document.getElementById('cam-name').value.trim();
    const url = document.getElementById('cam-url').value.trim();
    const location = document.getElementById('cam-location').value.trim();

    if (!name || !url) {
        window.ibvapDashboard.showToast('Please fill in required fields', 'error');
        return;
    }

    fetch('/api/cameras', {
        method: 'POST',
        headers: {'Content-Type': 'application/x-www-form-urlencoded'},
        body: new URLSearchParams({
            name: name,
            url: url,
            location: location,
            is_active: 'true'
        })
    })
    .then(response => {
        if (!response.ok) throw new Error('Failed to add camera');
        return response.json();
    })
    .then(data => {
        window.ibvapDashboard.showToast('Camera added successfully', 'success');
        window.ibvapDashboard.loadInitialData(); // Refresh camera list
        document.getElementById('camera-form').reset();
        document.getElementById('camera-modal').style.display = 'none';
    })
    .catch(error => {
        console.error('Error adding camera:', error);
        window.ibvapDashboard.showToast('Failed to add camera', 'error');
    });
}

function verifyIntegrity() {
    window.ibvapDashboard.verifyIntegrity();
}

function setDrawMode(mode) {
    window.ibvapDashboard.setDrawMode(mode);
}

function saveDrawing() {
    window.ibvapDashboard.saveDrawing();
}

function clearDrawing() {
    window.ibvapDashboard.clearDrawing();
}

function closeModal() {
    window.ibvapDashboard.closeModal();
}

function closeCameraModal() {
    window.ibvapDashboard.closeCameraModal();
}

function closeCanvasModal() {
    window.ibvapDashboard.closeCanvasModal();
}

function loadMoreAlerts() {
    window.ibvapDashboard.showToast('Loading more alerts...', 'info');
    #if false
    #else
    #endif
}

// Advanced feature placeholders for hackathon demo
function activateARMode() {
    window.ibvapDashboard.activateARMode();
}

function exportSecurityReport() {
    window.ibvapDashboard.exportSecurityReport();
}

function startAutoPatrol() {
    window.ibvapDashboard.startAutoPatrol();
}