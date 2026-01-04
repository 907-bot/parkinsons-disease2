/* ==================== Sample Data ====================*/
const SAMPLE_DATA = {
    'MDVP:Fo(Hz)': 119.992,
    'MDVP:Fhi(Hz)': 157.302,
    'MDVP:Flo(Hz)': 74.997,
    'MDVP:Jitter(%)': 0.00784,
    'MDVP:Jitter(Abs)': 0.00007,
    'MDVP:RAP': 0.00371,
    'MDVP:PPQ': 0.00554,
    'Jitter:DDP': 0.00112,
    'MDVP:Shimmer': 0.02182,
    'MDVP:Shimmer(dB)': 0.262,
    'Shimmer:APQ3': 0.01096,
    'Shimmer:APQ5': 0.01692,
    'MDVP:APQ': 0.02340,
    'Shimmer:DDA': 0.03287,
    'NHR': 0.02589,
    'HNR': 27.387,
    'status': 1,
    'RPDE': 0.521,
    'DFA': 0.665,
    'spread1': -5.363,
    'spread2': 0.283,
    'D2': 2.043,
    'PPE': 0.290
};

// List of all feature IDs
const ALL_FEATURES = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)',
    'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:RAP',
    'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer',
    'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'status', 'RPDE', 'DFA', 'spread1', 'spread2',
    'D2', 'PPE'
];

/* ==================== Tab Switching ====================*/
document.addEventListener('DOMContentLoaded', function() {
    // Tab switching functionality
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const tabName = this.getAttribute('data-tab');
            
            // Remove active class from all tabs and contents
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            // Add active class to clicked tab and corresponding content
            this.classList.add('active');
            document.getElementById(tabName).classList.add('active');
        });
    });

    // Initialize with first tab active
    if (tabBtns.length > 0) {
        tabBtns[0].classList.add('active');
    }
    if (tabContents.length > 0) {
        tabContents[0].classList.add('active');
    }

    // Mobile menu toggle
    setupMobileMenu();
});

/* ==================== Mobile Menu ====================*/
function setupMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');

    if (hamburger) {
        hamburger.addEventListener('click', function() {
            navMenu.style.display = navMenu.style.display === 'flex' ? 'none' : 'flex';
        });
    }
}

/* ==================== Load Sample Data ====================*/
function loadSampleData() {
    ALL_FEATURES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && SAMPLE_DATA[feature] !== undefined) {
            input.value = SAMPLE_DATA[feature];
        }
    });
    
    showNotification('Sample data loaded successfully!', 'success');
}

/* ==================== Clear Form ====================*/
function clearForm() {
    ALL_FEATURES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input) {
            input.value = '';
        }
    });
    
    // Hide results
    document.getElementById('results').style.display = 'none';
    document.getElementById('no-results').style.display = 'block';
    
    showNotification('Form cleared!', 'info');
}

/* ==================== Make Prediction ====================*/
async function makePrediction() {
    try {
        // Show loading state
        document.getElementById('loading').style.display = 'block';
        document.getElementById('results').style.display = 'none';

        // Collect all feature values
        const data = {};
        let isValid = true;

        ALL_FEATURES.forEach(feature => {
            const input = document.getElementById(feature);
            if (input) {
                const value = input.value.trim();
                
                if (!value) {
                    showNotification(`Missing value for: ${feature}`, 'error');
                    isValid = false;
                } else {
                    const numValue = parseFloat(value);
                    if (isNaN(numValue)) {
                        showNotification(`Invalid number for: ${feature}`, 'error');
                        isValid = false;
                    } else {
                        data[feature] = numValue;
                    }
                }
            }
        });

        if (!isValid) {
            document.getElementById('loading').style.display = 'none';
            return;
        }

        // Send to API
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        // Hide loading
        document.getElementById('loading').style.display = 'none';

        if (result.success) {
            displayResults(result);
            showNotification('Prediction completed!', 'success');
        } else {
            showNotification(`Error: ${result.error}`, 'error');
        }

    } catch (error) {
        console.error('Error:', error);
        document.getElementById('loading').style.display = 'none';
        showNotification('An error occurred during prediction', 'error');
    }
}

/* ==================== Display Results ====================*/
function displayResults(result) {
    const resultsDiv = document.getElementById('results');
    const noResultsDiv = document.getElementById('no-results');
    
    // Hide no results message
    noResultsDiv.style.display = 'none';

    // Determine status and styling
    const isParkinsons = result.prediction === 1;
    const statusIndicator = document.getElementById('status-indicator');
    
    if (isParkinsons) {
        statusIndicator.textContent = '⚠️';
        statusIndicator.className = 'status-indicator parkinsons';
    } else {
        statusIndicator.textContent = '✓';
        statusIndicator.className = 'status-indicator healthy';
    }

    // Update diagnosis
    document.getElementById('diagnosis-result').textContent = result.status;

    // Update confidence
    const confidence = result.confidence;
    const confidenceFill = document.getElementById('confidence-fill');
    confidenceFill.style.width = confidence + '%';
    document.getElementById('confidence-text').textContent = confidence.toFixed(1) + '%';

    // Update probabilities
    const probHealthy = result.probability_healthy * 100;
    const probParkinsons = result.probability_parkinsons * 100;

    const probHealthyEl = document.getElementById('prob-healthy');
    const probParkinsonsEl = document.getElementById('prob-parkinsons');

    probHealthyEl.style.width = probHealthy + '%';
    document.getElementById('prob-healthy-text').textContent = probHealthy.toFixed(1) + '%';

    probParkinsonsEl.style.width = probParkinsons + '%';
    document.getElementById('prob-parkinsons-text').textContent = probParkinsons.toFixed(1) + '%';

    // Update timestamp
    const timestamp = new Date(result.timestamp);
    document.getElementById('pred-time').textContent = timestamp.toLocaleString();

    // Show results
    resultsDiv.style.display = 'block';

    // Scroll to results
    setTimeout(() => {
        resultsDiv.scrollIntoView({ behavior: 'smooth' });
    }, 100);
}

/* ==================== Notification System ====================*/
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.innerHTML = `
        <div class="notification-content">
            <i class="fas fa-${getIconForType(type)}"></i>
            <p>${message}</p>
        </div>
    `;

    // Add to DOM
    document.body.appendChild(notification);

    // Trigger animation
    setTimeout(() => notification.classList.add('show'), 10);

    // Remove after 5 seconds
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 5000);
}

function getIconForType(type) {
    switch(type) {
        case 'success': return 'check-circle';
        case 'error': return 'exclamation-circle';
        case 'warning': return 'exclamation-triangle';
        default: return 'info-circle';
    }
}

/* ==================== Notification Styles (Dynamic) ====================*/
const style = document.createElement('style');
style.textContent = `
    .notification {
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 1rem 1.5rem;
        border-radius: 0.5rem;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        opacity: 0;
        transform: translateX(400px);
        transition: all 0.3s ease;
        z-index: 2000;
        max-width: 400px;
    }

    .notification.show {
        opacity: 1;
        transform: translateX(0);
    }

    .notification-content {
        display: flex;
        align-items: center;
        gap: 1rem;
    }

    .notification-success {
        border-left: 4px solid #27ae60;
    }

    .notification-success .notification-content i {
        color: #27ae60;
    }

    .notification-error {
        border-left: 4px solid #e74c3c;
    }

    .notification-error .notification-content i {
        color: #e74c3c;
    }

    .notification-warning {
        border-left: 4px solid #f39c12;
    }

    .notification-warning .notification-content i {
        color: #f39c12;
    }

    .notification-info {
        border-left: 4px solid #3498db;
    }

    .notification-info .notification-content i {
        color: #3498db;
    }

    .notification-content p {
        margin: 0;
        font-weight: 500;
    }

    @media (max-width: 600px) {
        .notification {
            right: 10px;
            left: 10px;
            max-width: none;
            transform: translateY(-400px);
        }

        .notification.show {
            transform: translateY(0);
        }
    }
`;
document.head.appendChild(style);

/* ==================== API Health Check ====================*/
async function checkAPIHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        console.log('API Status:', data);
        return data.success;
    } catch (error) {
        console.error('API Health Check Failed:', error);
        return false;
    }
}

// Check API on page load
window.addEventListener('load', async function() {
    const isHealthy = await checkAPIHealth();
    if (!isHealthy) {
        console.warn('API is not responding. Some features may not work.');
    }
});

/* ==================== Input Validation ====================*/
function validateInput(feature, value) {
    const numValue = parseFloat(value);
    
    if (isNaN(numValue)) {
        return { valid: false, message: `${feature} must be a number` };
    }

    // Feature-specific ranges (approximate)
    const ranges = {
        'MDVP:Fo(Hz)': { min: 0, max: 500 },
        'HNR': { min: 0, max: 50 },
        'RPDE': { min: 0, max: 1 },
        'DFA': { min: 0, max: 1 },
    };

    if (ranges[feature]) {
        const range = ranges[feature];
        if (numValue < range.min || numValue > range.max) {
            return { 
                valid: true, 
                message: `${feature} is outside typical range (${range.min}-${range.max}). Double-check your value.`,
                warning: true
            };
        }
    }

    return { valid: true };
}

/* ==================== Export Results ====================*/
function exportResults() {
    const resultsDiv = document.getElementById('results');
    if (resultsDiv.style.display === 'none') {
        showNotification('No results to export', 'warning');
        return;
    }

    // Collect data
    const data = {
        timestamp: new Date().toISOString(),
        diagnosis: document.getElementById('diagnosis-result').textContent,
        confidence: document.getElementById('confidence-text').textContent,
        features: {}
    };

    ALL_FEATURES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && input.value) {
            data.features[feature] = parseFloat(input.value);
        }
    });

    // Download as JSON
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(data, null, 2)));
    element.setAttribute('download', `parkinsons_prediction_${Date.now()}.json`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);

    showNotification('Results exported as JSON', 'success');
}

/* ==================== Keyboard Shortcuts ====================*/
document.addEventListener('keydown', function(event) {
    // Ctrl/Cmd + P for predict
    if ((event.ctrlKey || event.metaKey) && event.key === 'p') {
        event.preventDefault();
        makePrediction();
    }
    // Ctrl/Cmd + L to load sample
    if ((event.ctrlKey || event.metaKey) && event.key === 'l') {
        event.preventDefault();
        loadSampleData();
    }
});

/* ==================== Form Persistence ====================*/
function saveFormData() {
    const formData = {};
    ALL_FEATURES.forEach(feature => {
        const input = document.getElementById(feature);
        if (input && input.value) {
            formData[feature] = input.value;
        }
    });
    localStorage.setItem('parkinsons_form_data', JSON.stringify(formData));
}

function loadFormData() {
    const saved = localStorage.getItem('parkinsons_form_data');
    if (saved) {
        const formData = JSON.parse(saved);
        Object.keys(formData).forEach(feature => {
            const input = document.getElementById(feature);
            if (input) {
                input.value = formData[feature];
            }
        });
    }
}

// Auto-save form data on input change
document.addEventListener('input', function(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') {
        saveFormData();
    }
}, true);

// Load saved data on page load
window.addEventListener('load', loadFormData);

/* ==================== Utility Functions ====================*/
function formatNumber(num, decimals = 2) {
    return parseFloat(num).toFixed(decimals);
}

function getCurrentTimestamp() {
    return new Date().toISOString();
}