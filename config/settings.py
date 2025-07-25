"""
Configuration Settings for SafeData Pipeline
Centralized configuration management with environment support
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import logging

# Application metadata
APP_NAME = "SafeData Pipeline"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Privacy-Preserving Data Analysis Tool"
APP_AUTHOR = "SafeData Development Team"

# Environment configuration
ENVIRONMENT = os.getenv('SAFEDATA_ENV', 'production').lower()
DEBUG_MODE = ENVIRONMENT == 'development'

# Application paths
APP_HOME = Path.home() / ".safedata"
CONFIG_DIR = APP_HOME / "config"
DATA_DIR = APP_HOME / "data"
LOGS_DIR = APP_HOME / "logs"
CACHE_DIR = APP_HOME / "cache"
TEMP_DIR = APP_HOME / "temp"
REPORTS_DIR = APP_HOME / "reports"

# Ensure directories exist
for directory in [APP_HOME, CONFIG_DIR, DATA_DIR, LOGS_DIR, CACHE_DIR, TEMP_DIR, REPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Database configuration
DATABASE_CONFIG = {
    'path': str(DATA_DIR / "safedata.db"),
    'backup_enabled': True,
    'backup_interval_hours': 24,
    'cleanup_days': 30,
    'connection_timeout': 30
}

# Logging configuration
LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO' if not DEBUG_MODE else 'DEBUG'),
    'file_path': str(LOGS_DIR / "safedata.log"),
    'max_file_size_mb': 10,
    'backup_count': 5,
    'enable_console': DEBUG_MODE,
    'enable_file': True,
    'enable_audit': True,
    'audit_file_path': str(LOGS_DIR / "audit.log")
}

# GUI configuration
GUI_CONFIG = {
    'window_width': 1400,
    'window_height': 900,
    'min_width': 1200,
    'min_height': 800,
    'theme': os.getenv('SAFEDATA_THEME', 'dark'),
    'color_theme': os.getenv('SAFEDATA_COLOR_THEME', 'blue'),
    'font_family': 'Segoe UI',
    'font_size': 11,
    'dpi_scaling': 'auto',
    'auto_save_interval': 300,  # seconds
    'max_recent_files': 10
}

# Privacy analysis configuration
PRIVACY_CONFIG = {
    'default_epsilon': 1.0,
    'default_delta': 1e-5,
    'max_privacy_budget': 10.0,
    'default_k_anonymity': 3,
    'default_l_diversity': 2,
    'default_t_closeness': 0.2,
    'attack_simulation_samples': 1000,
    'quasi_identifier_detection': {
        'auto_detect': True,
        'uniqueness_threshold': 0.8,
        'min_cardinality': 2,
        'max_cardinality': 1000
    },
    'risk_thresholds': {
        'low': 0.1,
        'medium': 0.3,
        'high': 0.7,
        'critical': 0.9
    }
}

# Data processing configuration
DATA_CONFIG = {
    'max_file_size_mb': 500,
    'supported_formats': ['.csv', '.xlsx', '.xls', '.json', '.parquet', '.tsv'],
    'chunk_size': 10000,
    'memory_limit_mb': 1000,
    'encoding_detection': True,
    'auto_type_inference': True,
    'missing_value_patterns': ['', 'NULL', 'null', 'NaN', 'nan', 'N/A', 'n/a', '#N/A'],
    'datetime_formats': [
        '%Y-%m-%d', '%Y/%m/%d', '%d/%m/%Y', '%m/%d/%Y',
        '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S',
        '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S'
    ]
}

# Performance configuration
PERFORMANCE_CONFIG = {
    'max_workers': os.cpu_count() or 4,
    'enable_multiprocessing': True,
    'enable_caching': True,
    'cache_size_mb': 100,
    'progress_update_interval': 0.1,  # seconds
    'memory_monitoring': True,
    'performance_logging': DEBUG_MODE
}

# Security configuration
SECURITY_CONFIG = {
    'enable_encryption': True,
    'hash_algorithm': 'sha256',
    'secure_random_seed': True,
    'data_retention_days': 90,
    'audit_retention_days': 365,
    'session_timeout_minutes': 120,
    'max_login_attempts': 5,
    'password_policy': {
        'min_length': 8,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True
    }
}

# Report generation configuration
REPORTS_CONFIG = {
    'default_format': 'PDF',
    'supported_formats': ['PDF', 'HTML', 'JSON'],
    'output_directory': str(REPORTS_DIR),
    'template_directory': str(CONFIG_DIR / "templates"),
    'include_visualizations': True,
    'dpi': 150,
    'page_size': 'A4',
    'margins': {
        'top': 2.0,
        'bottom': 2.0,
        'left': 2.0,
        'right': 2.0
    },
    'max_chart_size': (800, 600),
    'watermark_enabled': False
}

# External services configuration
EXTERNAL_SERVICES_CONFIG = {
    'enable_internet_checks': False,
    'timeout_seconds': 30,
    'retry_attempts': 3,
    'proxy_enabled': False,
    'proxy_settings': {
        'http': None,
        'https': None
    }
}

# Development configuration
DEVELOPMENT_CONFIG = {
    'enable_hot_reload': DEBUG_MODE,
    'enable_profiling': DEBUG_MODE,
    'enable_memory_tracking': DEBUG_MODE,
    'mock_slow_operations': False,
    'test_data_enabled': DEBUG_MODE,
    'debug_gui': DEBUG_MODE
}

# Feature flags
FEATURE_FLAGS = {
    'advanced_privacy_techniques': True,
    'machine_learning_utility': True,
    'attack_simulation': True,
    'differential_privacy': True,
    'synthetic_data_generation': True,
    'real_time_monitoring': True,
    'batch_processing': True,
    'data_visualization': True,
    'export_capabilities': True,
    'user_management': False,  # Disabled for single-user application
    'cloud_integration': False,  # Disabled for local application
    'external_apis': False
}

# Compliance configuration
COMPLIANCE_CONFIG = {
    'dpdp_act_compliance': True,
    'gdpr_compliance': True,
    'hipaa_compliance': False,
    'audit_trail_enabled': True,
    'data_lineage_tracking': True,
    'consent_management': False,
    'right_to_erasure': True,
    'data_portability': True,
    'breach_notification': True
}

# Algorithm configuration
ALGORITHM_CONFIG = {
    'random_seed': 42,
    'numerical_precision': 1e-10,
    'convergence_tolerance': 1e-6,
    'max_iterations': 1000,
    'optimization_methods': ['gradient_descent', 'adam', 'lbfgs'],
    'default_optimizer': 'adam',
    'learning_rates': {
        'low': 0.001,
        'medium': 0.01,
        'high': 0.1
    }
}

class ConfigurationManager:
    """Manages application configuration with environment overrides and validation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_file = CONFIG_DIR / "user_config.json"
        self.user_config = {}
        self.load_user_config()
    
    def load_user_config(self):
        """Load user-specific configuration from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.user_config = json.load(f)
                self.logger.debug(f"Loaded user configuration from {self.config_file}")
        except Exception as e:
            self.logger.warning(f"Failed to load user configuration: {str(e)}")
            self.user_config = {}
    
    def save_user_config(self):
        """Save user-specific configuration to file"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_config, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved user configuration to {self.config_file}")
        except Exception as e:
            self.logger.error(f"Failed to save user configuration: {str(e)}")
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get configuration value with precedence: user_config > env_var > default_config"""
        
        # Check user configuration first
        if section in self.user_config and key in self.user_config[section]:
            return self.user_config[section][key]
        
        # Check environment variable
        env_key = f"SAFEDATA_{section.upper()}_{key.upper()}"
        env_value = os.getenv(env_key)
        if env_value is not None:
            # Try to parse as JSON, fallback to string
            try:
                return json.loads(env_value)
            except:
                return env_value
        
        # Get from default configuration
        config_map = {
            'database': DATABASE_CONFIG,
            'logging': LOGGING_CONFIG,
            'gui': GUI_CONFIG,
            'privacy': PRIVACY_CONFIG,
            'data': DATA_CONFIG,
            'performance': PERFORMANCE_CONFIG,
            'security': SECURITY_CONFIG,
            'reports': REPORTS_CONFIG,
            'external_services': EXTERNAL_SERVICES_CONFIG,
            'development': DEVELOPMENT_CONFIG,
            'features': FEATURE_FLAGS,
            'compliance': COMPLIANCE_CONFIG,
            'algorithms': ALGORITHM_CONFIG
        }
        
        if section in config_map and key in config_map[section]:
            return config_map[section][key]
        
        return default
    
    def set(self, section: str, key: str, value: Any):
        """Set user configuration value"""
        if section not in self.user_config:
            self.user_config[section] = {}
        
        self.user_config[section][key] = value
        self.save_user_config()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        config_map = {
            'database': DATABASE_CONFIG,
            'logging': LOGGING_CONFIG,
            'gui': GUI_CONFIG,
            'privacy': PRIVACY_CONFIG,
            'data': DATA_CONFIG,
            'performance': PERFORMANCE_CONFIG,
            'security': SECURITY_CONFIG,
            'reports': REPORTS_CONFIG,
            'external_services': EXTERNAL_SERVICES_CONFIG,
            'development': DEVELOPMENT_CONFIG,
            'features': FEATURE_FLAGS,
            'compliance': COMPLIANCE_CONFIG,
            'algorithms': ALGORITHM_CONFIG
        }
        
        base_config = config_map.get(section, {}).copy()
        
        # Override with user configuration
        if section in self.user_config:
            base_config.update(self.user_config[section])
        
        # Override with environment variables
        for key in base_config.keys():
            env_key = f"SAFEDATA_{section.upper()}_{key.upper()}"
            env_value = os.getenv(env_key)
            if env_value is not None:
                try:
                    base_config[key] = json.loads(env_value)
                except:
                    base_config[key] = env_value
        
        return base_config
    
    def reset_section(self, section: str):
        """Reset a configuration section to defaults"""
        if section in self.user_config:
            del self.user_config[section]
            self.save_user_config()
    
    def export_config(self, file_path: str):
        """Export current configuration to file"""
        try:
            config_export = {
                'app_metadata': {
                    'name': APP_NAME,
                    'version': APP_VERSION,
                    'environment': ENVIRONMENT
                },
                'database': self.get_section('database'),
                'logging': self.get_section('logging'),
                'gui': self.get_section('gui'),
                'privacy': self.get_section('privacy'),
                'data': self.get_section('data'),
                'performance': self.get_section('performance'),
                'security': self.get_section('security'),
                'reports': self.get_section('reports'),
                'features': self.get_section('features'),
                'compliance': self.get_section('compliance'),
                'algorithms': self.get_section('algorithms')
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_export, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration exported to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export configuration: {str(e)}")
            raise
    
    def import_config(self, file_path: str):
        """Import configuration from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                imported_config = json.load(f)
            
            # Validate and merge configuration
            for section in ['database', 'logging', 'gui', 'privacy', 'data', 
                          'performance', 'security', 'reports', 'features', 
                          'compliance', 'algorithms']:
                if section in imported_config:
                    self.user_config[section] = imported_config[section]
            
            self.save_user_config()
            self.logger.info(f"Configuration imported from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to import configuration: {str(e)}")
            raise
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate current configuration and return any issues"""
        issues = {}
        
        # Validate database configuration
        db_config = self.get_section('database')
        db_issues = []
        
        if not Path(db_config['path']).parent.exists():
            db_issues.append(f"Database directory does not exist: {Path(db_config['path']).parent}")
        
        if db_config['cleanup_days'] < 1:
            db_issues.append("Database cleanup days must be positive")
        
        if db_issues:
            issues['database'] = db_issues
        
        # Validate GUI configuration
        gui_config = self.get_section('gui')
        gui_issues = []
        
        if gui_config['window_width'] < 800:
            gui_issues.append("Window width too small (minimum 800)")
        
        if gui_config['window_height'] < 600:
            gui_issues.append("Window height too small (minimum 600)")
        
        if gui_issues:
            issues['gui'] = gui_issues
        
        # Validate privacy configuration
        privacy_config = self.get_section('privacy')
        privacy_issues = []
        
        if privacy_config['default_epsilon'] <= 0:
            privacy_issues.append("Default epsilon must be positive")
        
        if privacy_config['default_delta'] <= 0 or privacy_config['default_delta'] >= 1:
            privacy_issues.append("Default delta must be between 0 and 1")
        
        if privacy_issues:
            issues['privacy'] = privacy_issues
        
        return issues

# Global configuration manager instance
config_manager = ConfigurationManager()

# Convenience functions for common configuration access
def get_database_config() -> Dict[str, Any]:
    """Get database configuration"""
    return config_manager.get_section('database')

def get_gui_config() -> Dict[str, Any]:
    """Get GUI configuration"""
    return config_manager.get_section('gui')

def get_privacy_config() -> Dict[str, Any]:
    """Get privacy analysis configuration"""
    return config_manager.get_section('privacy')

def get_logging_config() -> Dict[str, Any]:
    """Get logging configuration"""
    return config_manager.get_section('logging')

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration"""
    return config_manager.get_section('performance')

def get_feature_flags() -> Dict[str, bool]:
    """Get feature flags"""
    return config_manager.get_section('features')

def is_debug_mode() -> bool:
    """Check if debug mode is enabled"""
    return DEBUG_MODE

def get_app_info() -> Dict[str, str]:
    """Get application metadata"""
    return {
        'name': APP_NAME,
        'version': APP_VERSION,
        'description': APP_DESCRIPTION,
        'author': APP_AUTHOR,
        'environment': ENVIRONMENT
    }

def get_app_directories() -> Dict[str, Path]:
    """Get application directories"""
    return {
        'home': APP_HOME,
        'config': CONFIG_DIR,
        'data': DATA_DIR,
        'logs': LOGS_DIR,
        'cache': CACHE_DIR,
        'temp': TEMP_DIR,
        'reports': REPORTS_DIR
    }

# Configuration for common application settings
APP_CONFIG = {
    'name': APP_NAME,
    'version': APP_VERSION,
    'description': APP_DESCRIPTION,
    'author': APP_AUTHOR,
    'environment': ENVIRONMENT,
    'debug_mode': DEBUG_MODE,
    'directories': get_app_directories(),
    'database': get_database_config(),
    'gui': get_gui_config(),
    'privacy': get_privacy_config(),
    'logging': get_logging_config(),
    'performance': get_performance_config(),
    'features': get_feature_flags()
}
