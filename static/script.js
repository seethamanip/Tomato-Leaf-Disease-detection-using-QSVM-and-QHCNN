// API Configuration
const API_BASE_URL = 'http://localhost:5000';

// DOM Elements
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const loadingSpinner = document.getElementById('loadingSpinner');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Event Listeners
imageInput.addEventListener('change', handleImageSelect);

// Drag and drop functionality
document.addEventListener('dragover', (e) => {
    e.preventDefault();
    document.querySelector('.upload-box').style.borderColor = '#764ba2';
});

document.addEventListener('dragleave', (e) => {
    e.preventDefault();
    document.querySelector('.upload-box').style.borderColor = '#667eea';
});

document.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        imageInput.files = files;
        handleImageSelect();
    }
});

/**
 * Handle image selection
 */
async function handleImageSelect() {
    const file = imageInput.files[0];
    
    if (!file) return;
    
    // Validate file type
    if (!['image/jpeg', 'image/png', 'image/jpg'].includes(file.type)) {
        showError('Invalid file type. Please upload JPG or PNG.');
        return;
    }
    
    // Validate file size (8MB)
    if (file.size > 8 * 1024 * 1024) {
        showError('File is too large. Maximum size is 8MB.');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        // Clear any previous prediction values immediately so UI shows a neutral state
        clearResultsUI();
        errorSection.classList.add('hidden');
    };
    reader.readAsDataURL(file);
    
    // Send prediction request
    await sendPredictionRequest(file);
}

/**
 * Send image to server for prediction
 */
async function sendPredictionRequest(file) {
    try {
        // Show loading spinner
        loadingSpinner.classList.remove('hidden');
        resultsSection.classList.add('hidden');
        
        // Create FormData
        const formData = new FormData();
        formData.append('image', file);
        
        // Send request
        const response = await fetch(`${API_BASE_URL}/api/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }
        
        const result = await response.json();
        displayResults(result);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to process image. Please try again.');
    } finally {
        loadingSpinner.classList.add('hidden');
    }
}

/**
 * Display prediction results
 */
function displayResults(result) {
    // Disease name
    document.getElementById('diseaseName').textContent = result.disease_name;
    
    // Confidence score with percentage
    const confidencePercent = (result.confidence_score * 100).toFixed(2);
    document.getElementById('confidenceScore').textContent = 
        `${result.confidence_score.toFixed(4)} (${confidencePercent}%)`;
    
    // Model accuracy
    if (result.accuracy_percent) {
        document.getElementById('modelAccuracy').textContent = 
            `${result.accuracy_percent.toFixed(2)}%`;
    } else {
        document.getElementById('modelAccuracy').textContent = 'Not available';
    }
    
    // Per-class similarities
    displaySimilarityChart(result.per_class_similarity);
    
    // Show results section
    resultsSection.classList.remove('hidden');
    errorSection.classList.add('hidden');
}

/**
 * Display similarity chart
 */
function displaySimilarityChart(similarities) {
    const chartContainer = document.getElementById('similarityChart');
    chartContainer.innerHTML = '';
    
    // Sort by similarity score (descending)
    const sorted = Object.entries(similarities)
        .sort((a, b) => b[1] - a[1]);
    
    sorted.forEach(([className, score]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'similarity-bar';
        
        const labelDiv = document.createElement('div');
        labelDiv.className = 'similarity-label';
        labelDiv.textContent = className;
        
        const containerDiv = document.createElement('div');
        containerDiv.className = 'similarity-bar-container';
        
        const fillDiv = document.createElement('div');
        fillDiv.className = 'similarity-bar-fill';
        fillDiv.style.width = (score * 100) + '%';
        fillDiv.textContent = (score * 100).toFixed(1) + '%';
        
        containerDiv.appendChild(fillDiv);
        barDiv.appendChild(labelDiv);
        barDiv.appendChild(containerDiv);
        chartContainer.appendChild(barDiv);
    });
}

/**
 * Reset form for next prediction
 */
function resetForm() {
    imageInput.value = '';
    previewSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');
    loadingSpinner.classList.add('hidden');
    // Clear previous results UI when resetting
    clearResultsUI();
}

/**
 * Clear result fields to a neutral/blank state.
 */
function clearResultsUI() {
    const dn = document.getElementById('diseaseName');
    const cs = document.getElementById('confidenceScore');
    const ma = document.getElementById('modelAccuracy');
    const chart = document.getElementById('similarityChart');
    if (dn) dn.textContent = '';
    if (cs) cs.textContent = '';
    if (ma) ma.textContent = '';
    if (chart) chart.innerHTML = '';
}

/**
 * Show error message
 */
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    errorSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    loadingSpinner.classList.add('hidden');
}

/**
 * Close error message
 */
function closeError() {
    errorSection.classList.add('hidden');
}

// Check server health on page load
document.addEventListener('DOMContentLoaded', async () => {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        if (!response.ok) {
            console.warn('Server health check failed');
        }
    } catch (error) {
        console.warn('Server connection error:', error);
        showError('⚠️ Could not connect to server. Make sure the server is running on port 5000.');
    }
    // Load comparison summary (if any)
    loadComparison();
});

// Comparison panel logic
document.getElementById('refreshComparison')?.addEventListener('click', () => {
    loadComparison(true);
});

async function loadComparison(force = false) {
    const container = document.getElementById('comparisonContent');
    const downloadLink = document.getElementById('downloadComparison');
    container.innerHTML = '<p class="muted">Loading comparison summary...</p>';
    downloadLink.style.display = 'none';
    try {
        const resp = await fetch(`${API_BASE_URL}/api/comparison`);
        if (!resp.ok) {
            const err = await resp.json();
            container.innerHTML = `<p class="muted">${err.error || 'Comparison not available.'}</p>`;
            return;
        }
        const ct = resp.headers.get('content-type') || '';
        if (ct.includes('application/json')) {
            const data = await resp.json();
            // if file contains model_comparison structure
            if (data.model_comparison) {
                renderComparisonFromJson(data.model_comparison, container);
                // enable download link to the raw json served by server
                downloadLink.href = `${API_BASE_URL}/artifacts/model_comparison.json`;
                downloadLink.style.display = 'inline-block';
            } else if (data.report_text) {
                container.innerHTML = `<pre class="comparison-pre">${escapeHtml(data.report_text)}</pre>`;
                downloadLink.href = '#';
            } else {
                // show keys/summary
                container.innerHTML = `<pre class="comparison-pre">${escapeHtml(JSON.stringify(data, null, 2))}</pre>`;
            }
        } else {
            // fallback: show text body
            const text = await resp.text();
            container.innerHTML = `<pre class="comparison-pre">${escapeHtml(text)}</pre>`;
        }
    } catch (e) {
        container.innerHTML = `<p class="muted">Failed to load comparison: ${e.message}</p>`;
    }
}

function renderComparisonFromJson(obj, container) {
    // Build a visual HTML table with bars for accuracy
    container.innerHTML = '';
    const table = document.createElement('table');
    table.className = 'comparison-table';

    // Header
    const thead = document.createElement('thead');
    thead.innerHTML = `
        <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>F1-Score</th>
            <th>Inference (s)</th>
            <th>Training (s)</th>
        </tr>`;
    table.appendChild(thead);

    const tbody = document.createElement('tbody');
    // Sort by accuracy desc if available
    const rows = Object.entries(obj).sort((a,b)=> (b[1].accuracy||0)-(a[1].accuracy||0));
    for (const [model, metrics] of rows) {
        const tr = document.createElement('tr');
        const acc = (metrics.accuracy || 0);
        const prec = (metrics.precision || 0);
        const rec = (metrics.recall || 0);
        const f1 = (metrics.f1_score || 0);
        const inf = (metrics.inference_time_seconds ?? metrics.inference_time ?? 'N/A');
        const train = (metrics.training_time_seconds ?? metrics.training_time ?? 'N/A');

        tr.innerHTML = `
            <td class="model-name">${escapeHtml(model)}</td>
            <td class="accuracy-cell">
                <div class="bar">
                    <div class="bar-fill" style="width:${(acc*100).toFixed(2)}%">${(acc*100).toFixed(1)}%</div>
                </div>
            </td>
            <td>${(prec*100).toFixed(2)}%</td>
            <td>${(rec*100).toFixed(2)}%</td>
            <td>${(f1*100).toFixed(2)}%</td>
            <td>${typeof inf === 'number' ? inf.toFixed(4) : escapeHtml(String(inf))}</td>
            <td>${typeof train === 'number' ? train.toFixed(4) : escapeHtml(String(train))}</td>
        `;
        tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    container.appendChild(table);
}

function escapeHtml(s) {
    return s.replace(/[&<>"']/g, function (c) {
        return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":"&#39;"}[c];
    });
}
