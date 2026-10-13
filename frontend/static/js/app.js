/**
 * ═══════════════════════════════════════════════════════════════════════════
 * MEDICAL AI DIAGNOSIS SUPPORT SYSTEM — FRONTEND LOGIC
 * Handles image upload, API communication, result rendering, and UI state
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ─── State Management ──────────────────────────────────────────────────────
const AppState = {
    selectedFile: null,
    isAnalyzing: false,
    currentResult: null,
    userLocation: { lat: 28.6139, lng: 77.2090 }  // Default: New Delhi
};

// ─── DOM Elements ──────────────────────────────────────────────────────────
const DOM = {
    fileInput: () => document.getElementById('fileInput'),
    uploadZone: () => document.getElementById('uploadZone'),
    previewContainer: () => document.getElementById('previewContainer'),
    previewPlaceholder: () => document.getElementById('previewPlaceholder'),
    analyzeBtn: () => document.getElementById('analyzeBtn'),
    resultsSection: () => document.getElementById('resultsSection'),
    loadingOverlay: () => document.getElementById('loadingOverlay'),
    toastContainer: () => document.getElementById('toastContainer')
};

// ─── Initialization ────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initUploadHandlers();
    initGeolocation();
    console.log('🏥 Medical AI System initialized');
});

// ─── Upload Handlers ───────────────────────────────────────────────────────
function initUploadHandlers() {
    const uploadZone = DOM.uploadZone();
    const fileInput = DOM.fileInput();

    // Click to upload
    uploadZone.addEventListener('click', () => fileInput.click());

    // File selected
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    // Drag & Drop
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
}

// ─── File Selection ────────────────────────────────────────────────────────
function handleFileSelect(file) {
    const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/tiff'];

    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.dcm')) {
        showToast('error', 'Invalid file type. Please upload PNG, JPG, BMP, TIFF, or DICOM files.');
        return;
    }

    if (file.size > 16 * 1024 * 1024) {
        showToast('error', 'File too large. Maximum size is 16MB.');
        return;
    }

    AppState.selectedFile = file;
    updatePreview(file);
    DOM.uploadZone().classList.add('has-file');
    DOM.analyzeBtn().disabled = false;
    showToast('success', `Image loaded: ${file.name}`);
}

// ─── Image Preview ─────────────────────────────────────────────────────────
function updatePreview(file) {
    const container = DOM.previewContainer();
    const reader = new FileReader();

    reader.onload = (e) => {
        const sizeKB = (file.size / 1024).toFixed(1);
        const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
        const displaySize = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;

        container.innerHTML = `
            <div class="preview-image-container">
                <img src="${e.target.result}" alt="Medical Image Preview" class="preview-image" id="previewImage">
                <div class="preview-info">
                    <span class="preview-badge">📄 ${file.name}</span>
                    <span class="preview-badge">📏 ${displaySize}</span>
                    <span class="preview-badge">🖼️ ${file.type || 'DICOM'}</span>
                </div>
            </div>
        `;
    };

    reader.readAsDataURL(file);
}

// ─── Analysis ──────────────────────────────────────────────────────────────
async function analyzeImage() {
    if (!AppState.selectedFile || AppState.isAnalyzing) return;

    AppState.isAnalyzing = true;
    DOM.analyzeBtn().disabled = true;
    showLoading(true);

    const formData = new FormData();
    formData.append('image', AppState.selectedFile);
    formData.append('latitude', AppState.userLocation.lat);
    formData.append('longitude', AppState.userLocation.lng);

    // Animate loading steps
    animateLoadingSteps();

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            AppState.currentResult = data;
            renderResults(data);
            showToast('success', `Analysis complete: ${data.prediction.disease_title}`);
        } else {
            throw new Error(data.error || 'Analysis failed');
        }
    } catch (error) {
        console.error('Analysis error:', error);
        showToast('error', `Analysis failed: ${error.message}`);
    } finally {
        AppState.isAnalyzing = false;
        DOM.analyzeBtn().disabled = false;
        showLoading(false);
    }
}

// ─── Loading Animation ────────────────────────────────────────────────────
function showLoading(show) {
    const overlay = DOM.loadingOverlay();
    if (show) {
        overlay.classList.add('active');
    } else {
        overlay.classList.remove('active');
    }
}

function animateLoadingSteps() {
    const steps = document.querySelectorAll('.loading-step');
    const delays = [0, 800, 1600, 2400, 3200];

    steps.forEach((step, index) => {
        setTimeout(() => {
            steps.forEach(s => s.classList.remove('active'));
            step.classList.add('active');
            if (index > 0) steps[index - 1].classList.add('done');
        }, delays[index]);
    });
}

// ─── Render Results ────────────────────────────────────────────────────────
function renderResults(data) {
    const section = DOM.resultsSection();
    section.innerHTML = '';

    // Prediction Hero Card
    section.innerHTML += renderPredictionHero(data);

    // Results Grid (Grad-CAM + Disease Info)
    section.innerHTML += renderResultsGrid(data);

    // Health Suggestions
    section.innerHTML += renderHealthSuggestions(data);

    // Hospital Recommendations
    section.innerHTML += renderHospitals(data);

    // Metadata
    section.innerHTML += renderMetadata(data);

    section.classList.add('visible');

    // Animate confidence circle
    setTimeout(() => animateConfidenceCircle(data.prediction.confidence), 100);

    // Animate probability bars
    setTimeout(() => animateProbabilityBars(data.prediction.all_probabilities), 300);

    // Scroll to results
    section.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Prediction Hero ───────────────────────────────────────────────────────
function renderPredictionHero(data) {
    const { prediction, disease_info } = data;
    const riskClass = prediction.risk_level.level.toLowerCase().replace(/[- ]/g, '-');
    const circumference = 2 * Math.PI * 45;

    return `
        <div class="result-hero risk-${riskClass}">
            <div class="section-header">
                <div class="section-icon blue">🔬</div>
                <div>
                    <div class="section-title">AI Diagnosis Result</div>
                    <div class="section-subtitle">Deep learning analysis powered by MobileNetV2</div>
                </div>
            </div>

            <div class="result-hero-grid">
                <div class="result-icon-wrap risk-${riskClass}">
                    ${disease_info.icon}
                </div>
                <div>
                    <div class="result-disease-name">${prediction.disease_title}</div>
                    <div class="result-disease-desc">${disease_info.description.substring(0, 150)}...</div>
                    <div class="risk-badge risk-${riskClass}">
                        ⚡ Risk Level: ${prediction.risk_level.level}
                    </div>
                </div>
                <div class="result-confidence-wrap">
                    <div class="confidence-circle">
                        <svg viewBox="0 0 100 100">
                            <circle class="confidence-track" cx="50" cy="50" r="45"/>
                            <circle class="confidence-fill ${getConfidenceClass(prediction.confidence)}"
                                    cx="50" cy="50" r="45"
                                    stroke-dasharray="${circumference}"
                                    stroke-dashoffset="${circumference}"
                                    id="confidenceCircle"/>
                        </svg>
                        <div class="confidence-text">
                            <span class="confidence-value" id="confidenceValue">0%</span>
                            <span class="confidence-label">Confidence</span>
                        </div>
                    </div>
                </div>
            </div>

            <div class="prob-bars">
                ${renderProbabilityBars(prediction.all_probabilities)}
            </div>
        </div>
    `;
}

function getConfidenceClass(confidence) {
    if (confidence >= 70) return 'high';
    if (confidence >= 40) return 'medium';
    return 'low';
}

function renderProbabilityBars(probs) {
    return Object.entries(probs)
        .sort(([, a], [, b]) => b - a)
        .map(([name, value]) => `
            <div class="prob-bar-item">
                <span class="prob-bar-label">${name}</span>
                <div class="prob-bar-track">
                    <div class="prob-bar-fill ${name}" style="width: 0%" data-target="${value}"></div>
                </div>
                <span class="prob-bar-value" data-target="${value}">0%</span>
            </div>
        `).join('');
}

function animateConfidenceCircle(confidence) {
    const circle = document.getElementById('confidenceCircle');
    const valueEl = document.getElementById('confidenceValue');
    if (!circle || !valueEl) return;

    const circumference = 2 * Math.PI * 45;
    const offset = circumference - (confidence / 100) * circumference;

    circle.style.strokeDashoffset = offset;

    // Animate number
    let current = 0;
    const increment = confidence / 40;
    const timer = setInterval(() => {
        current += increment;
        if (current >= confidence) {
            current = confidence;
            clearInterval(timer);
        }
        valueEl.textContent = `${Math.round(current)}%`;
    }, 25);
}

function animateProbabilityBars(probs) {
    document.querySelectorAll('.prob-bar-fill').forEach(bar => {
        const target = bar.dataset.target;
        bar.style.width = `${target}%`;
    });

    document.querySelectorAll('.prob-bar-value').forEach(val => {
        const target = parseFloat(val.dataset.target);
        let current = 0;
        const increment = target / 30;
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            val.textContent = `${current.toFixed(1)}%`;
        }, 30);
    });
}

// ─── Results Grid ──────────────────────────────────────────────────────────
function renderResultsGrid(data) {
    return `
        <div class="results-grid">
            ${renderGradCAMCard(data)}
            ${renderDiseaseInfoCard(data)}
        </div>
    `;
}

// ─── Grad-CAM Card ─────────────────────────────────────────────────────────
function renderGradCAMCard(data) {
    return `
        <div class="card gradcam-card">
            <div class="card-header">
                <div class="section-header" style="margin-bottom:0">
                    <div class="section-icon red">🔥</div>
                    <div>
                        <div class="section-title">Explainable AI — Grad-CAM</div>
                        <div class="section-subtitle">Visual explanation of AI decision</div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <img src="${data.explainability.gradcam_url}" alt="Grad-CAM Heatmap" class="gradcam-image"
                     onerror="this.src='${data.explainability.gradcam_overlay_url}'; this.onerror=null;">
                <div class="gradcam-explanation">
                    <strong>🤖 AI Explanation:</strong><br>
                    ${data.explainability.explanation}
                </div>
            </div>
        </div>
    `;
}

// ─── Disease Info Card ─────────────────────────────────────────────────────
function renderDiseaseInfoCard(data) {
    const info = data.disease_info;

    return `
        <div class="card disease-info-card">
            <div class="card-header">
                <div class="section-header" style="margin-bottom:0">
                    <div class="section-icon purple">📋</div>
                    <div>
                        <div class="section-title">Medical Information</div>
                        <div class="section-subtitle">Comprehensive disease knowledge</div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="info-section">
                    <div class="info-section-title">⚕️ Recommended Specialist</div>
                    <div style="padding:8px 14px; background:var(--primary-50); border-radius:8px; font-weight:600; color:var(--primary-700); font-size:0.88rem;">
                        ${info.specialist}
                    </div>
                </div>

                <div class="info-section">
                    <div class="info-section-title">🔍 Causes</div>
                    <ul class="info-list causes">
                        ${info.causes.map(c => `<li>${c}</li>`).join('')}
                    </ul>
                </div>

                <div class="info-section">
                    <div class="info-section-title">🩺 Symptoms</div>
                    <ul class="info-list symptoms">
                        ${info.symptoms.map(s => `<li>${s}</li>`).join('')}
                    </ul>
                </div>

                <div class="info-section">
                    <div class="info-section-title">⚠️ Risk Factors</div>
                    <ul class="info-list causes">
                        ${info.risk_factors.map(r => `<li>${r}</li>`).join('')}
                    </ul>
                </div>

                <div class="info-section">
                    <div class="info-section-title">🛡️ Prevention Tips</div>
                    <ul class="info-list prevention">
                        ${info.prevention.map(p => `<li>${p}</li>`).join('')}
                    </ul>
                </div>

                <div class="info-section">
                    <div class="info-section-title">💊 Precautions</div>
                    <ul class="info-list precautions">
                        ${info.precautions.map(p => `<li>${p}</li>`).join('')}
                    </ul>
                </div>
            </div>
        </div>
    `;
}

// ─── Health Suggestions ────────────────────────────────────────────────────
function renderHealthSuggestions(data) {
    const suggestions = data.health_suggestions;

    return `
        <div class="card suggestions-card">
            <div class="card-header">
                <div class="section-header" style="margin-bottom:0">
                    <div class="section-icon green">💡</div>
                    <div>
                        <div class="section-title">Smart Health Suggestions</div>
                        <div class="section-subtitle">AI-generated health recommendations</div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="suggestions-grid">
                    ${suggestions.map(s => {
                        const icon = s.match(/^([\u{1F300}-\u{1FAD6}\u{2600}-\u{26FF}\u{2700}-\u{27BF}⚕️⚠️✅❄️🚫])/u);
                        const text = icon ? s.substring(icon[0].length).trim() : s;
                        const displayIcon = icon ? icon[0] : '💊';
                        return `
                            <div class="suggestion-item">
                                <span class="suggestion-icon">${displayIcon}</span>
                                <span>${text}</span>
                            </div>
                        `;
                    }).join('')}
                </div>
            </div>
        </div>
    `;
}

// ─── Hospital Recommendations ──────────────────────────────────────────────
function renderHospitals(data) {
    const hospitals = data.hospitals;

    return `
        <div class="card hospitals-card">
            <div class="card-header">
                <div class="section-header" style="margin-bottom:0">
                    <div class="section-icon teal">🏥</div>
                    <div>
                        <div class="section-title">Nearby Hospital Recommendations</div>
                        <div class="section-subtitle">Specialist hospitals for ${data.prediction.disease_title}</div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="hospitals-grid">
                    ${hospitals.map(h => renderHospitalCard(h)).join('')}
                </div>
            </div>
        </div>
    `;
}

function renderHospitalCard(hospital) {
    const stars = '★'.repeat(Math.floor(hospital.rating)) +
                  (hospital.rating % 1 >= 0.5 ? '½' : '');

    return `
        <div class="hospital-item">
            <div class="hospital-name">${hospital.name}</div>
            <span class="hospital-type">${hospital.type}</span>
            <div class="hospital-details">
                <div class="hospital-detail">
                    <span class="hospital-detail-icon">📍</span>
                    <span>${hospital.address}, ${hospital.city}</span>
                </div>
                <div class="hospital-detail">
                    <span class="hospital-detail-icon">📏</span>
                    <span>${hospital.distance_km} km away</span>
                </div>
                <div class="hospital-detail">
                    <span class="hospital-detail-icon">📞</span>
                    <span>${hospital.phone}</span>
                </div>
                <div class="hospital-detail">
                    <span class="hospital-detail-icon">⭐</span>
                    <span class="hospital-rating">
                        <span class="star">${stars}</span>
                        <span>${hospital.rating}/5</span>
                    </span>
                </div>
                ${hospital.emergency ? `
                <div class="hospital-detail">
                    <span class="hospital-detail-icon">🚑</span>
                    <span style="color: #e53935; font-weight: 600;">24/7 Emergency Available</span>
                </div>` : ''}
            </div>
            <div class="hospital-specialties">
                ${hospital.specialties.map(s => {
                    const isMatch = hospital.matching_specialties &&
                                    hospital.matching_specialties.includes(s);
                    return `<span class="specialty-tag ${isMatch ? 'match' : ''}">${s}</span>`;
                }).join('')}
            </div>
        </div>
    `;
}

// ─── Metadata Card ─────────────────────────────────────────────────────────
function renderMetadata(data) {
    const meta = data.metadata;
    const timestamp = new Date(meta.timestamp).toLocaleString();

    return `
        <div class="card metadata-card">
            <div class="card-header">
                <div class="section-header" style="margin-bottom:0">
                    <div class="section-icon orange">📊</div>
                    <div>
                        <div class="section-title">Analysis Metadata</div>
                        <div class="section-subtitle">Technical details of this analysis</div>
                    </div>
                </div>
            </div>
            <div class="card-body">
                <div class="metadata-grid">
                    <div class="metadata-item">
                        <div class="metadata-label">Analysis ID</div>
                        <div class="metadata-value">${meta.file_id}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Original File</div>
                        <div class="metadata-value">${meta.original_filename}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Processing Time</div>
                        <div class="metadata-value">${meta.processing_time_seconds}s</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Model Version</div>
                        <div class="metadata-value">${meta.model_version}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Timestamp</div>
                        <div class="metadata-value">${timestamp}</div>
                    </div>
                    <div class="metadata-item">
                        <div class="metadata-label">Urgency Level</div>
                        <div class="metadata-value" style="color: ${data.disease_info.urgency === 'critical' ? '#e53935' : data.disease_info.urgency === 'high' ? '#f57c00' : '#4caf50'}">
                            ${data.disease_info.urgency.toUpperCase()}
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// ─── Geolocation ───────────────────────────────────────────────────────────
function initGeolocation() {
    if ('geolocation' in navigator) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                AppState.userLocation = {
                    lat: position.coords.latitude,
                    lng: position.coords.longitude
                };
                console.log('📍 User location detected:', AppState.userLocation);
            },
            (error) => {
                console.log('📍 Using default location (Delhi)');
            },
            { timeout: 5000 }
        );
    }
}

// ─── Toast Notifications ───────────────────────────────────────────────────
function showToast(type, message) {
    const container = DOM.toastContainer();
    const icons = { success: '✅', error: '❌', info: 'ℹ️' };

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span>${icons[type] || 'ℹ️'}</span><span>${message}</span>`;

    container.appendChild(toast);

    setTimeout(() => {
        if (toast.parentNode) toast.parentNode.removeChild(toast);
    }, 3500);
}

// ─── Smooth Scroll ─────────────────────────────────────────────────────────
function scrollToSection(id) {
    const el = document.getElementById(id);
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ─── Reset / New Analysis ──────────────────────────────────────────────────
function resetAnalysis() {
    AppState.selectedFile = null;
    AppState.currentResult = null;

    DOM.fileInput().value = '';
    DOM.uploadZone().classList.remove('has-file');
    DOM.analyzeBtn().disabled = true;

    const container = DOM.previewContainer();
    container.innerHTML = `
        <div class="preview-placeholder" id="previewPlaceholder">
            <div class="preview-placeholder-icon">🖼️</div>
            <p>Image preview will appear here</p>
        </div>
    `;

    DOM.resultsSection().classList.remove('visible');
    DOM.resultsSection().innerHTML = '';

    showToast('info', 'Ready for new analysis');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}
