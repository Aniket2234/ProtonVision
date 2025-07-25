"""
SafeData Pipeline - Web Application Entry Point
Privacy-Preserving Data Analysis Tool for Replit
"""

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import logging
from pathlib import Path
import json
import io
import base64

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from modules.data_loader import DataLoader
from modules.risk_assessment import RiskAssessment
from modules.privacy_enhancement import PrivacyEnhancement
from modules.utility_measurement import UtilityMeasurement
from reports.report_generator import ReportGenerator
from utils.logger import setup_logging
from utils.database import DatabaseManager
from config.settings import get_app_info, get_gui_config

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')
CORS(app)

# Configure Flask
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

class SafeDataWebApp:
    """Web application wrapper for SafeData Pipeline"""
    
    def __init__(self):
        self.setup_application()
        self.setup_modules()
        
    def setup_application(self):
        """Setup application configuration and logging"""
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.db_manager = DatabaseManager()
        self.db_manager.initialize_database()
        
        self.logger.info("SafeData Pipeline web application initialized")
        
    def setup_modules(self):
        """Initialize core processing modules"""
        self.data_loader = DataLoader()
        self.risk_assessment = RiskAssessment()
        self.privacy_enhancement = PrivacyEnhancement()
        self.utility_measurement = UtilityMeasurement()
        self.report_generator = ReportGenerator()

# Global app instance
safedata_app = SafeDataWebApp()

@app.route('/')
def index():
    """Main dashboard page"""
    app_info = get_app_info()
    return render_template('index.html', app_info=app_info)

@app.route('/help')
def help_page():
    """Help and usage guide page"""
    return render_template('help.html')

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'app_info': get_app_info(),
        'modules_loaded': True
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Handle file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
            
        # Save uploaded file temporarily
        temp_path = Path('temp') / file.filename
        temp_path.parent.mkdir(exist_ok=True)
        file.save(str(temp_path))
        
        # Load and validate data
        result = safedata_app.data_loader.load_file(str(temp_path))
        
        if result['success']:
            # Ensure all data is JSON serializable
            data_info = result['data_info']
            safe_data_info = {
                'filename': str(data_info.get('filename', '')),
                'rows': int(data_info.get('rows', 0)),
                'columns': int(data_info.get('columns', 0)),
                'size': int(data_info.get('size', 0)),
                'format': str(data_info.get('format', '')),
                'column_names': [str(col) for col in data_info.get('column_names', [])],
                'data_types': {str(k): str(v) for k, v in data_info.get('data_types', {}).items()},
                'memory_usage': int(data_info.get('memory_usage', 0)),
                'sample_data': data_info.get('sample_data', [])
            }
            
            return jsonify({
                'success': True,
                'message': 'File uploaded successfully',
                'data_info': safe_data_info
            })
        else:
            return jsonify({
                'error': result['error']
            }), 400
            
    except Exception as e:
        safedata_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """Perform privacy analysis"""
    try:
        data = request.get_json()
        analysis_type = data.get('type', 'risk_assessment')
        parameters = data.get('parameters', {})
        
        if analysis_type == 'risk_assessment':
            result = safedata_app.risk_assessment.assess_privacy_risk(
                dataset=safedata_app.data_loader.current_dataset,
                **parameters
            )
        elif analysis_type == 'privacy_enhancement':
            result = safedata_app.privacy_enhancement.apply_techniques(
                dataset=safedata_app.data_loader.current_dataset,
                **parameters
            )
        elif analysis_type == 'utility_measurement':
            result = safedata_app.utility_measurement.measure_utility(
                original_dataset=safedata_app.data_loader.current_dataset,
                **parameters
            )
        else:
            return jsonify({'error': 'Unknown analysis type'}), 400
            
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        safedata_app.logger.error(f"Analysis error: {str(e)}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/report', methods=['POST'])
def api_report():
    """Generate and download report"""
    try:
        data = request.get_json()
        report_format = data.get('format', 'PDF')
        
        # Generate report
        report_path = safedata_app.report_generator.generate_report(
            format=report_format,
            data=data.get('data', {}),
            include_visualizations=True
        )
        
        if report_path and Path(report_path).exists():
            return send_file(
                report_path,
                as_attachment=True,
                download_name=f"safedata_report.{report_format.lower()}"
            )
        else:
            return jsonify({'error': 'Report generation failed'}), 500
            
    except Exception as e:
        safedata_app.logger.error(f"Report generation error: {str(e)}")
        return jsonify({'error': f'Report generation failed: {str(e)}'}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    print(f"Starting SafeData Pipeline on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)