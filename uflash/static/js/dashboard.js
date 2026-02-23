/**
 * U-Flash Mobile Dashboard
 * Handles tab navigation, API calls, and Chart.js visualizations
 */

// ===== Tab Navigation =====
function switchTab(tab) {
    // Hide all screens
    document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
    // Deactivate all nav items
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));

    // Show target screen
    const screen = document.getElementById('screen' + tab.charAt(0).toUpperCase() + tab.slice(1));
    if (screen) {
        screen.classList.add('active');
    }

    // Activate nav item
    const navItem = document.querySelector(`.nav-item[data-tab="${tab}"]`);
    if (navItem) {
        navItem.classList.add('active');
    }
}

// ===== Slider Updates =====
const distSlider = document.getElementById('distSlider');
const depthSlider = document.getElementById('depthSlider');
const luxSlider = document.getElementById('luxSlider');
const distVal = document.getElementById('distVal');
const depthVal = document.getElementById('depthVal');
const luxVal = document.getElementById('luxVal');

if (distSlider) distSlider.oninput = function () {
    distVal.textContent = parseFloat(this.value).toFixed(1) + 'm';
    updateSettings();
};
if (depthSlider) depthSlider.oninput = function () {
    depthVal.textContent = parseFloat(this.value).toFixed(1) + 'm';
    updateSettings();
};
if (luxSlider) luxSlider.oninput = function () {
    luxVal.textContent = parseInt(this.value) + ' lux';
    updateSettings();
};

function updateSettings() {
    const settDist = document.getElementById('settDistance');
    const settDepth = document.getElementById('settDepth');
    const settLux = document.getElementById('settLux');
    if (settDist) settDist.textContent = parseFloat(distSlider.value).toFixed(1) + ' m';
    if (settDepth) settDepth.textContent = parseFloat(depthSlider.value).toFixed(1) + ' m';
    if (settLux) settLux.textContent = parseInt(luxSlider.value) + ' lux';
}

// ===== Water Type Selection =====
document.querySelectorAll('.select-btn[data-param="water_type"]').forEach(btn => {
    btn.addEventListener('click', function () {
        document.querySelectorAll('.select-btn[data-param="water_type"]').forEach(b => b.classList.remove('active'));
        this.classList.add('active');
        const settWt = document.getElementById('settWaterType');
        if (settWt) settWt.textContent = this.dataset.value;
    });
});

// ===== Chart.js Configuration =====
const chartColors = {
    cyan: '#00d4ff',
    cyanAlpha: 'rgba(0, 212, 255, 0.15)',
    teal: '#00e5a0',
    tealAlpha: 'rgba(0, 229, 160, 0.15)',
    rose: '#f43f5e',
    grid: 'rgba(255, 255, 255, 0.05)',
    text: '#7aa5c4',
};

Chart.defaults.color = chartColors.text;
Chart.defaults.borderColor = chartColors.grid;
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 11;

let waveformChart = null;
let berChart = null;

function initWaveformChart() {
    const ctx = document.getElementById('waveformChart');
    if (!ctx) return;
    if (waveformChart) waveformChart.destroy();

    waveformChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Transmitted',
                    data: [],
                    borderColor: chartColors.cyan,
                    backgroundColor: chartColors.cyanAlpha,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1,
                },
                {
                    label: 'Received',
                    data: [],
                    borderColor: chartColors.teal,
                    backgroundColor: chartColors.tealAlpha,
                    borderWidth: 1.5,
                    pointRadius: 0,
                    fill: true,
                    tension: 0.1,
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600 },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { boxWidth: 12, padding: 8, font: { size: 10 } }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: { display: true, text: 'Sample', font: { size: 10 } },
                    grid: { display: false },
                    ticks: { maxTicksLimit: 6, font: { size: 9 } }
                },
                y: {
                    display: true,
                    title: { display: true, text: 'Amplitude', font: { size: 10 } },
                    grid: { color: chartColors.grid },
                    ticks: { font: { size: 9 } }
                }
            }
        }
    });
}

function initBerChart() {
    const ctx = document.getElementById('berChart');
    if (!ctx) return;
    if (berChart) berChart.destroy();

    berChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BER',
                data: [],
                borderColor: chartColors.rose,
                backgroundColor: 'rgba(244, 63, 94, 0.1)',
                borderWidth: 2,
                pointRadius: 4,
                pointBackgroundColor: chartColors.rose,
                fill: true,
                tension: 0.3,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 600 },
            plugins: {
                legend: {
                    display: true,
                    position: 'top',
                    labels: { boxWidth: 12, padding: 8, font: { size: 10 } }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Distance (m)', font: { size: 10 } },
                    grid: { display: false },
                    ticks: { font: { size: 9 } }
                },
                y: {
                    title: { display: true, text: 'BER', font: { size: 10 } },
                    grid: { color: chartColors.grid },
                    ticks: { font: { size: 9 } },
                    type: 'logarithmic',
                    min: 1e-7,
                }
            }
        }
    });
}

// ===== Transmit =====
const transmitBtn = document.getElementById('transmitBtn');
if (transmitBtn) {
    transmitBtn.addEventListener('click', async function () {
        const btn = this;
        btn.classList.add('loading');
        btn.innerHTML = '<div class="spinner"></div> Transmitting...';

        const headerStatus = document.getElementById('headerStatus');
        if (headerStatus) headerStatus.innerHTML = '<span class="status-dot" style="background:var(--amber)"></span><span>Transmitting...</span>';

        const params = {
            message: document.getElementById('messageInput').value || 'Hello Underwater World!',
            water_type: document.querySelector('.select-btn.active[data-param="water_type"]')?.dataset.value || 'clear',
            distance_m: parseFloat(distSlider.value),
            depth_m: parseFloat(depthSlider.value),
            ambient_lux: parseFloat(luxSlider.value),
            motion_level: document.getElementById('motionSelect').value,
        };

        try {
            const resp = await fetch('/api/transmit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params),
            });
            const data = await resp.json();
            displayResults(data);
            updatePipeline(data);

            if (headerStatus) headerStatus.innerHTML = '<span class="status-dot"></span><span>Complete</span>';

            // Auto-switch to results
            setTimeout(() => switchTab('results'), 500);
        } catch (err) {
            console.error('Transmit error:', err);
            if (headerStatus) headerStatus.innerHTML = '<span class="status-dot" style="background:var(--rose)"></span><span>Error</span>';
        }

        btn.classList.remove('loading');
        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/></svg> Transmit';
    });
}

// ===== BER Sweep =====
const sweepBtn = document.getElementById('sweepBtn');
if (sweepBtn) {
    sweepBtn.addEventListener('click', async function () {
        const btn = this;
        btn.innerHTML = '<div class="spinner"></div> Running Sweep...';

        try {
            const resp = await fetch('/api/ber_sweep', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message: document.getElementById('messageInput').value || 'Hello',
                    distances: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                }),
            });
            const data = await resp.json();
            displayBerSweep(data);
            setTimeout(() => switchTab('results'), 500);
        } catch (err) {
            console.error('BER sweep error:', err);
        }

        btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 3v18h18"/><path d="M18 9l-5 5-4-4-3 3"/></svg> Run BER Sweep';
    });
}

// ===== Display Results =====
function displayResults(data) {
    // Hide empty state
    const emptyState = document.getElementById('resultsEmpty');
    if (emptyState) emptyState.style.display = 'none';

    // Result banner
    const banner = document.getElementById('resultBanner');
    const icon = document.getElementById('resultIcon');
    const label = document.getElementById('resultLabel');
    const detail = document.getElementById('resultDetail');

    if (banner) {
        banner.classList.remove('hidden', 'success', 'warning', 'error');
        const berFloat = data.ber_float || 0;
        const berStr = data.ber || '0.00e+00';
        if (berFloat === 0) {
            banner.classList.add('success');
            icon.textContent = '✓';
            label.textContent = 'Perfect Transmission';
        } else if (berFloat < 0.01) {
            banner.classList.add('success');
            icon.textContent = '✓';
            label.textContent = 'Transmission Successful';
        } else if (berFloat < 0.1) {
            banner.classList.add('warning');
            icon.textContent = '⚠';
            label.textContent = 'Partial Errors Detected';
        } else {
            banner.classList.add('error');
            icon.textContent = '✕';
            label.textContent = 'High Error Rate';
        }
        detail.textContent = `BER: ${berStr} | ${data.bit_errors || 0} errors | ${(data.elapsed_ms || 0).toFixed(1)}ms`;
    }

    // Decoded text
    const decoded = document.getElementById('decodedText');
    if (decoded) decoded.textContent = data.decoded_message || '—';

    // Metrics
    const chMetrics = data.stages?.channel || {};
    setValue('metricBER', data.ber || '—');
    setValue('metricSNR', chMetrics.snr_db != null ? Number(chMetrics.snr_db).toFixed(1) : '—');
    setValue('metricFEC', data.fec_mode || '—');
    setValue('metricRate', '180');
    setValue('metricQuality', data.channel_quality || '—');
    setValue('metricLatency', data.elapsed_ms != null ? data.elapsed_ms.toFixed(1) : '—');

    // Waveform chart
    const txWave = data.waveforms?.tx;
    const rxWave = data.waveforms?.rx;
    if (txWave && rxWave) {
        initWaveformChart();
        const maxPts = 200;
        const step = Math.max(1, Math.floor(txWave.length / maxPts));
        const labels = [];
        const txData = [];
        const rxData = [];
        for (let i = 0; i < txWave.length; i += step) {
            labels.push(i);
            txData.push(txWave[i]);
            rxData.push(rxWave[i]);
        }
        waveformChart.data.labels = labels;
        waveformChart.data.datasets[0].data = txData;
        waveformChart.data.datasets[1].data = rxData;
        waveformChart.update();
    }
}

function setValue(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}

// ===== Update Pipeline =====
function updatePipeline(data) {
    const stages = data.stages || {};

    updateStageFromObj('rll', stages.rll, s => `${s.config || ''} • ${s.overhead || ''}`);
    updateStageFromObj('conv', stages.convolutional, s => s.skipped ? 'Skipped' : `Rate ${s.rate || '1/2'} • K=${s.constraint_length || 7}`);
    updateStageFromObj('int', stages.interleaver, s => s.skipped ? 'Skipped' : `Depth=${s.depth || 20} • Burst: ${s.burst_protection || '?'}`);
    updateStageFromObj('ppm', stages.modulation, s => `${s.type || '4-PPM'} • ${s.waveform_samples || 0} samples`);
    updateStageFromObj('ch', stages.channel, s => `SNR: ${Number(s.snr_db || 0).toFixed(1)}dB • BER: ${Number(s.raw_ber || 0).toExponential(1)}`);
    updateStageFromObj('motion', stages.motion, s => `Avg AngVel: ${Number(s.avg_angular_velocity || 0).toFixed(1)}°/s`);
    updateStageFromObj('wq', stages.water_quality, s => `${s.water_type || '?'} • NTU: ${Number(s.ntu_estimate || 0).toFixed(1)}`);
    updateStageFromObj('ml', stages.channel_prediction, s => s.status || 'Active');
}

function updateStageFromObj(prefix, info, detailFn) {
    if (!info) return;
    const badge = document.getElementById(prefix + 'Badge');
    const detail = document.getElementById(prefix + 'Detail');
    if (badge) badge.textContent = '✓';
    if (detail) {
        try { detail.textContent = detailFn(info); }
        catch (e) { detail.textContent = JSON.stringify(info).slice(0, 80); }
    }
}

// ===== Display BER Sweep =====
function displayBerSweep(data) {
    const emptyState = document.getElementById('resultsEmpty');
    if (emptyState) emptyState.style.display = 'none';

    const container = document.getElementById('sweepContainer');
    if (container) container.style.display = 'block';

    initBerChart();
    if (data.sweep_results && data.sweep_results.length > 0) {
        berChart.data.labels = data.sweep_results.map(r => r.distance_m + 'm');
        berChart.data.datasets[0].data = data.sweep_results.map(r => Math.max(r.avg_ber, 1e-7));
        berChart.update();
    }
}

// ===== Initialize on Load =====
document.addEventListener('DOMContentLoaded', function () {
    initWaveformChart();
    updateSettings();
});
