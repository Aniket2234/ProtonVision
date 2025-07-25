// SafeData Pipeline Web Interface JavaScript

class SafeDataApp {
    constructor() {
        this.currentDataset = null;
        this.currentAnalysis = null;
        this.setupEventListeners();
        this.checkStatus();
    }

    setupEventListeners() {
        // File upload
        document.getElementById('upload-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handleFileUpload();
        });

        // Drag and drop functionality
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');

        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, this.preventDefaults, false);
            document.body.addEventListener(eventName, this.preventDefaults, false);
        });

        // Highlight drop area when item is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragover'), false);
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragover'), false);
        });

        // Handle dropped files
        uploadArea.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                this.updateFileDisplay(files[0]);
                this.handleFileUpload();
            }
        });

        // Handle file input change
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.updateFileDisplay(e.target.files[0]);
            }
        });

        // Analysis buttons
        document.getElementById('btn-risk-assessment').addEventListener('click', () => {
            this.selectAnalysis('risk_assessment');
        });
        
        document.getElementById('btn-privacy-enhancement').addEventListener('click', () => {
            this.selectAnalysis('privacy_enhancement');
        });
        
        document.getElementById('btn-utility-measurement').addEventListener('click', () => {
            this.selectAnalysis('utility_measurement');
        });

        // Run analysis
        document.getElementById('btn-run-analysis').addEventListener('click', () => {
            this.runAnalysis();
        });

        // Generate report
        document.getElementById('btn-generate-report').addEventListener('click', () => {
            this.generateReport();
        });
    }

    async checkStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            if (data.status === 'running') {
                this.showStatus('System ready for analysis', 'success');
            }
        } catch (error) {
            this.showStatus('Failed to connect to server', 'danger');
        }
    }

    async handleFileUpload() {
        const fileInput = document.getElementById('file-input');
        
        if (!fileInput) {
            this.showStatus('File input not found', 'danger');
            return;
        }
        
        if (!fileInput.files || fileInput.files.length === 0) {
            this.showStatus('Please select a file', 'warning');
            return;
        }
        
        const file = fileInput.files[0];

        const formData = new FormData();
        formData.append('file', file);

        this.showLoading('Uploading and processing file...');
        this.showProgress(0);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            this.hideLoading();

            if (response.ok) {
                const data = await response.json();
                this.currentDataset = data.data_info;
                this.showDataInfo(data.data_info);
                this.enableAnalysisButtons();
                this.showStatus('File uploaded successfully', 'success');
            } else {
                const error = await response.json();
                this.showStatus(error.error || 'Upload failed', 'danger');
            }
        } catch (error) {
            this.hideLoading();
            this.showStatus('Upload failed: ' + error.message, 'danger');
        }
    }

    showDataInfo(dataInfo) {
        const container = document.getElementById('data-details');
        const infoDiv = document.getElementById('data-info');
        
        container.innerHTML = `
            <div class="data-info-item">
                <span class="data-info-label">Filename:</span>
                <span class="data-info-value">${dataInfo.filename}</span>
            </div>
            <div class="data-info-item">
                <span class="data-info-label">Rows:</span>
                <span class="data-info-value">${dataInfo.rows.toLocaleString()}</span>
            </div>
            <div class="data-info-item">
                <span class="data-info-label">Columns:</span>
                <span class="data-info-value">${dataInfo.columns.toLocaleString()}</span>
            </div>
            <div class="data-info-item">
                <span class="data-info-label">Size:</span>
                <span class="data-info-value">${this.formatBytes(dataInfo.size)}</span>
            </div>
            <div class="data-info-item">
                <span class="data-info-label">Format:</span>
                <span class="data-info-value">${dataInfo.format}</span>
            </div>
        `;
        
        infoDiv.classList.remove('d-none');
        infoDiv.classList.add('fade-in');
    }

    enableAnalysisButtons() {
        const buttons = [
            'btn-risk-assessment',
            'btn-privacy-enhancement', 
            'btn-utility-measurement'
        ];
        
        buttons.forEach(id => {
            document.getElementById(id).disabled = false;
        });
    }

    selectAnalysis(type) {
        this.currentAnalysis = type;
        this.showAnalysisParams(type);
        
        // Update button states
        document.querySelectorAll('#analysis-options .btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.getElementById(`btn-${type.replace('_', '-')}`).classList.add('active');
    }

    showAnalysisParams(type) {
        const container = document.getElementById('param-controls');
        const paramsDiv = document.getElementById('analysis-params');
        
        let paramsHTML = '';
        
        switch (type) {
            case 'risk_assessment':
                paramsHTML = `
                    <div class="param-group">
                        <label class="param-label">Attack Simulation Samples</label>
                        <div class="param-control">
                            <input type="number" class="form-control param-input" 
                                   id="param-samples" value="1000" min="100" max="10000">
                        </div>
                        <div class="param-info">Number of samples for attack simulation</div>
                    </div>
                    <div class="param-group">
                        <label class="param-label">Quasi-Identifier Detection</label>
                        <div class="param-control">
                            <select class="form-select" id="param-qi-detection">
                                <option value="auto">Automatic</option>
                                <option value="manual">Manual Selection</option>
                            </select>
                        </div>
                    </div>
                `;
                break;
                
            case 'privacy_enhancement':
                paramsHTML = `
                    <div class="param-group">
                        <label class="param-label">Privacy Technique</label>
                        <div class="param-control">
                            <select class="form-select" id="param-technique">
                                <option value="differential_privacy">Differential Privacy</option>
                                <option value="statistical_disclosure">Statistical Disclosure Control</option>
                                <option value="synthetic_data">Synthetic Data Generation</option>
                            </select>
                        </div>
                    </div>
                    <div class="param-group">
                        <label class="param-label">Privacy Budget (ε)</label>
                        <div class="param-control">
                            <input type="number" class="form-control param-input" 
                                   id="param-epsilon" value="1.0" min="0.1" max="10" step="0.1">
                        </div>
                        <div class="param-info">Lower values provide stronger privacy</div>
                    </div>
                `;
                break;
                
            case 'utility_measurement':
                paramsHTML = `
                    <div class="param-group">
                        <label class="param-label">Utility Metrics</label>
                        <div class="param-control">
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="metric-statistical" checked>
                                <label class="form-check-label" for="metric-statistical">
                                    Statistical Utility
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" id="metric-ml" checked>
                                <label class="form-check-label" for="metric-ml">
                                    Machine Learning Utility
                                </label>
                            </div>
                        </div>
                    </div>
                `;
                break;
        }
        
        container.innerHTML = paramsHTML;
        paramsDiv.classList.remove('d-none');
        paramsDiv.classList.add('fade-in');
    }

    async runAnalysis() {
        if (!this.currentAnalysis) {
            this.showStatus('Please select an analysis type', 'warning');
            return;
        }

        const parameters = this.collectParameters();
        
        this.showLoading('Running analysis...');
        
        try {
            const response = await fetch('/api/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    type: this.currentAnalysis,
                    parameters: parameters
                })
            });

            this.hideLoading();

            if (response.ok) {
                const data = await response.json();
                this.showResults(data.result);
                this.showStatus('Analysis completed successfully', 'success');
            } else {
                const error = await response.json();
                this.showStatus(error.error || 'Analysis failed', 'danger');
            }
        } catch (error) {
            this.hideLoading();
            this.showStatus('Analysis failed: ' + error.message, 'danger');
        }
    }

    collectParameters() {
        const parameters = {};
        
        switch (this.currentAnalysis) {
            case 'risk_assessment':
                parameters.samples = parseInt(document.getElementById('param-samples')?.value || 1000);
                parameters.qi_detection = document.getElementById('param-qi-detection')?.value || 'auto';
                break;
                
            case 'privacy_enhancement':
                parameters.technique = document.getElementById('param-technique')?.value || 'differential_privacy';
                parameters.epsilon = parseFloat(document.getElementById('param-epsilon')?.value || 1.0);
                break;
                
            case 'utility_measurement':
                parameters.metrics = {
                    statistical: document.getElementById('metric-statistical')?.checked || false,
                    ml: document.getElementById('metric-ml')?.checked || false
                };
                break;
        }
        
        return parameters;
    }

    showResults(results) {
        const container = document.getElementById('results-content');
        const section = document.getElementById('results-section');
        
        let resultsHTML = '';
        
        if (this.currentAnalysis === 'risk_assessment') {
            resultsHTML = this.formatRiskResults(results);
        } else if (this.currentAnalysis === 'privacy_enhancement') {
            resultsHTML = this.formatPrivacyResults(results);
        } else if (this.currentAnalysis === 'utility_measurement') {
            resultsHTML = this.formatUtilityResults(results);
        }
        
        container.innerHTML = resultsHTML;
        section.style.display = 'block';
        section.classList.add('slide-up');
    }

    formatRiskResults(results) {
        const riskLevel = this.getRiskLevel(results.overall_risk || 0);
        
        return `
            <div class="result-metric ${riskLevel}">
                <span class="metric-label">Overall Privacy Risk</span>
                <span class="metric-value">${(results.overall_risk * 100).toFixed(1)}%</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">K-Anonymity Score</span>
                <span class="metric-value">${results.k_anonymity || 'N/A'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">L-Diversity Score</span>
                <span class="metric-value">${results.l_diversity || 'N/A'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Vulnerable Records</span>
                <span class="metric-value">${results.vulnerable_records || 0}</span>
            </div>
        `;
    }

    formatPrivacyResults(results) {
        return `
            <div class="result-metric">
                <span class="metric-label">Privacy Technique Applied</span>
                <span class="metric-value">${results.technique || 'Unknown'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Privacy Budget Used</span>
                <span class="metric-value">ε = ${results.epsilon_used || 'N/A'}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Records Protected</span>
                <span class="metric-value">${results.records_protected || 0}</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Data Quality Retained</span>
                <span class="metric-value">${(results.quality_retained * 100).toFixed(1)}%</span>
            </div>
        `;
    }

    formatUtilityResults(results) {
        return `
            <div class="result-metric">
                <span class="metric-label">Statistical Utility</span>
                <span class="metric-value">${(results.statistical_utility * 100).toFixed(1)}%</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">ML Model Accuracy</span>
                <span class="metric-value">${(results.ml_accuracy * 100).toFixed(1)}%</span>
            </div>
            <div class="result-metric">
                <span class="metric-label">Correlation Preservation</span>
                <span class="metric-value">${(results.correlation_preservation * 100).toFixed(1)}%</span>
            </div>
        `;
    }

    getRiskLevel(risk) {
        if (risk >= 0.7) return 'high-risk';
        if (risk >= 0.3) return 'medium-risk';
        return 'low-risk';
    }

    async generateReport() {
        this.showLoading('Generating report...');
        
        try {
            const response = await fetch('/api/report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    format: 'PDF',
                    data: {
                        dataset: this.currentDataset,
                        analysis: this.currentAnalysis,
                        results: this.collectResults()
                    }
                })
            });

            this.hideLoading();

            if (response.ok) {
                // Trigger download
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'safedata_report.pdf';
                a.click();
                window.URL.revokeObjectURL(url);
                
                this.showStatus('Report downloaded successfully', 'success');
            } else {
                const error = await response.json();
                this.showStatus(error.error || 'Report generation failed', 'danger');
            }
        } catch (error) {
            this.hideLoading();
            this.showStatus('Report generation failed: ' + error.message, 'danger');
        }
    }

    collectResults() {
        // Collect current results from the UI
        const resultsContainer = document.getElementById('results-content');
        return {
            html: resultsContainer.innerHTML,
            timestamp: new Date().toISOString()
        };
    }

    showStatus(message, type) {
        const alert = document.getElementById('status-alert');
        const messageSpan = document.getElementById('status-message');
        
        alert.className = `alert alert-${type}`;
        messageSpan.textContent = message;
        alert.classList.remove('d-none');
        
        setTimeout(() => {
            alert.classList.add('d-none');
        }, 5000);
    }

    showLoading(message) {
        const modal = new bootstrap.Modal(document.getElementById('loading-modal'));
        document.getElementById('loading-message').textContent = message;
        modal.show();
    }

    hideLoading() {
        const modal = bootstrap.Modal.getInstance(document.getElementById('loading-modal'));
        if (modal) {
            modal.hide();
        }
    }

    showProgress(percent) {
        const progressBar = document.querySelector('#upload-progress .progress-bar');
        const progressDiv = document.getElementById('upload-progress');
        
        if (percent > 0) {
            progressDiv.classList.remove('d-none');
            progressBar.style.width = percent + '%';
        } else {
            progressDiv.classList.add('d-none');
        }
    }

    preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    updateFileDisplay(file) {
        const uploadArea = document.getElementById('upload-area');
        uploadArea.innerHTML = `
            <div class="file-selected">
                <div class="file-name">
                    <i class="fas fa-file me-2"></i>${file.name}
                </div>
                <div class="file-details mt-2">
                    <small>Size: ${this.formatBytes(file.size)} | Type: ${file.type || 'Unknown'}</small>
                </div>
            </div>
        `;
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.safeDataApp = new SafeDataApp();
});