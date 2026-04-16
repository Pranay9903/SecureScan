// Main JavaScript file for SecureScan

document.addEventListener('DOMContentLoaded', function() {
    initAlerts();
    initTooltips();
});

function initAlerts() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(alert => {
        setTimeout(() => {
            alert.style.opacity = '0';
            setTimeout(() => alert.remove(), 300);
        }, 5000);
    });
}

function initTooltips() {
    const tooltips = document.querySelectorAll('[data-tooltip]');
    tooltips.forEach(el => {
        el.addEventListener('mouseenter', () => {
            const tooltip = document.createElement('div');
            tooltip.className = 'tooltip';
            tooltip.textContent = el.dataset.tooltip;
            document.body.appendChild(tooltip);
            
            const rect = el.getBoundingClientRect();
            tooltip.style.left = rect.left + rect.width / 2 - tooltip.offsetWidth / 2 + 'px';
            tooltip.style.top = rect.top - tooltip.offsetHeight - 10 + 'px';
            
            el._tooltip = tooltip;
        });
        
        el.addEventListener('mouseleave', () => {
            if (el._tooltip) {
                el._tooltip.remove();
                delete el._tooltip;
            }
        });
    });
}

async function scanUrl(url, scanType = 'quick') {
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    
    if (!url) {
        showError('Please enter a URL');
        return;
    }
    
    if (!url.startsWith('http://') && !url.startsWith('https://')) {
        url = 'https://' + url;
    }
    
    try {
        const response = await fetch(scanType === 'quick' ? '/detect' : '/analyze', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({url: url, scan_type: scanType})
        });
        
        const data = await response.json();
        
        if (data.error) {
            showError(data.error);
        } else {
            displayResult(data);
        }
    } catch (e) {
        showError('Connection error. Please try again.');
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const isPhishing = data.prediction === 'Phishing';
    
    resultDiv.innerHTML = `
        <div class="result-card ${isPhishing ? 'phishing' : 'safe'}">
            <div class="result-icon">
                <i class="fas fa-${isPhishing ? 'exclamation-triangle' : 'check-circle'}"></i>
            </div>
            <h3>${data.prediction}</h3>
            <p class="confidence">Confidence: ${(data.confidence * 100).toFixed(1)}%</p>
            
            ${data.url_features ? `
                <div class="url-features">
                    <h4>URL Analysis</h4>
                    <div class="features-list">
                        <div class="feature">
                            <span class="label">URL Length</span>
                            <span class="value">${data.url_features.url_length || 'N/A'}</span>
                        </div>
                        <div class="feature">
                            <span class="label">Has HTTPS</span>
                            <span class="value">${data.url_features.has_https ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="feature">
                            <span class="label">Contains IP</span>
                            <span class="value">${data.url_features.has_ip ? 'Yes' : 'No'}</span>
                        </div>
                        <div class="feature">
                            <span class="label">Suspicious TLD</span>
                            <span class="value">${data.url_features.suspicious_tld ? 'Yes' : 'No'}</span>
                        </div>
                    </div>
                </div>
            ` : ''}
            
            ${data.individual_predictions && Object.keys(data.individual_predictions).length > 0 ? `
                <div class="model-predictions">
                    <h4>Model Predictions</h4>
                    ${Object.entries(data.individual_predictions).map(([name, pred]) => `
                        <div class="model-bar">
                            <span class="label">${name.toUpperCase()}</span>
                            <div class="bar">
                                <div class="bar-fill" style="width: ${pred.probability * 100}%"></div>
                            </div>
                            <span class="value">${(pred.probability * 100).toFixed(1)}%</span>
                        </div>
                    `).join('')}
                </div>
            ` : ''}
            
            <div class="result-actions">
                <button class="btn btn-outline" onclick="saveScan()">
                    <i class="fas fa-save"></i> Save Scan
                </button>
                <button class="btn btn-outline" onclick="shareResult()">
                    <i class="fas fa-share"></i> Share
                </button>
            </div>
        </div>
    `;
}

function showError(message) {
    const resultDiv = document.getElementById('result');
    resultDiv.innerHTML = `
        <div class="alert alert-error">
            <i class="fas fa-exclamation-circle"></i> ${message}
        </div>
    `;
}

function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString() + ' ' + date.toLocaleTimeString();
}

function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        alert('Copied to clipboard!');
    });
}

function exportToCSV() {
    window.location.href = '/export/history';
}

function filterScans(status) {
    const rows = document.querySelectorAll('.scan-row');
    rows.forEach(row => {
        if (status === 'all' || row.dataset.status === status) {
            row.style.display = '';
        } else {
            row.style.display = 'none';
        }
    });
}

function toggleSidebar() {
    document.querySelector('.sidebar').classList.toggle('collapsed');
}