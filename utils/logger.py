"""
Logging Configuration Module for SafeData Pipeline
Sets up comprehensive logging with multiple handlers and formatters
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import os

def setup_logging(log_level: str = "INFO", 
                 log_file: Optional[str] = None,
                 enable_console: bool = True,
                 enable_file: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5) -> logging.Logger:
    """
    Setup comprehensive logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (default: ~/.safedata/logs/safedata.log)
        enable_console: Enable console logging
        enable_file: Enable file logging
        max_file_size: Maximum log file size before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    if log_file is None:
        log_dir = Path.home() / ".safedata" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "safedata.log"
    else:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if enable_file:
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_file),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
    
    # Error file handler (separate file for errors)
    if enable_file:
        error_log_file = log_file.parent / "safedata_errors.log"
        error_handler = logging.handlers.RotatingFileHandler(
            filename=str(error_log_file),
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(error_handler)
    
    # Setup specific loggers for different modules
    setup_module_loggers(log_level)
    
    # Log initial message
    logger = logging.getLogger(__name__)
    logger.info("SafeData Pipeline logging initialized")
    logger.info(f"Log level: {log_level}")
    logger.info(f"Log file: {log_file}")
    
    return root_logger

def setup_module_loggers(log_level: str = "INFO"):
    """Setup specific loggers for different application modules"""
    
    # GUI module logger
    gui_logger = logging.getLogger('gui')
    gui_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Modules logger
    modules_logger = logging.getLogger('modules')
    modules_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Utils logger
    utils_logger = logging.getLogger('utils')
    utils_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Reports logger
    reports_logger = logging.getLogger('reports')
    reports_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Database logger
    db_logger = logging.getLogger('database')
    db_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Privacy logger (for sensitive operations)
    privacy_logger = logging.getLogger('privacy')
    privacy_logger.setLevel(getattr(logging, log_level.upper()))

class SafeDataLogger:
    """Custom logger class for SafeData Pipeline with additional features"""
    
    def __init__(self, name: str, session_id: Optional[str] = None):
        self.logger = logging.getLogger(name)
        self.session_id = session_id
        self.db_manager = None
        
    def set_session_id(self, session_id: str):
        """Set session ID for logging context"""
        self.session_id = session_id
    
    def set_database_manager(self, db_manager):
        """Set database manager for logging to database"""
        self.db_manager = db_manager
    
    def _log_to_database(self, level: str, message: str, user_action: Optional[str] = None):
        """Log message to database if database manager is available"""
        if self.db_manager:
            try:
                self.db_manager.log_application_event(
                    level=level,
                    module=self.logger.name,
                    message=message,
                    session_id=self.session_id,
                    user_action=user_action
                )
            except Exception:
                # Don't fail if database logging fails
                pass
    
    def debug(self, message: str, user_action: Optional[str] = None):
        """Log debug message"""
        self.logger.debug(message)
        self._log_to_database('DEBUG', message, user_action)
    
    def info(self, message: str, user_action: Optional[str] = None):
        """Log info message"""
        self.logger.info(message)
        self._log_to_database('INFO', message, user_action)
    
    def warning(self, message: str, user_action: Optional[str] = None):
        """Log warning message"""
        self.logger.warning(message)
        self._log_to_database('WARNING', message, user_action)
    
    def error(self, message: str, user_action: Optional[str] = None):
        """Log error message"""
        self.logger.error(message)
        self._log_to_database('ERROR', message, user_action)
    
    def critical(self, message: str, user_action: Optional[str] = None):
        """Log critical message"""
        self.logger.critical(message)
        self._log_to_database('CRITICAL', message, user_action)
    
    def log_user_action(self, action: str, details: Optional[str] = None):
        """Log user action with optional details"""
        message = f"User action: {action}"
        if details:
            message += f" - {details}"
        
        self.info(message, user_action=action)
    
    def log_privacy_operation(self, operation: str, dataset_info: str, 
                            parameters: Optional[str] = None):
        """Log privacy-sensitive operations"""
        message = f"Privacy operation: {operation} on {dataset_info}"
        if parameters:
            message += f" with parameters: {parameters}"
        
        # Use privacy logger for sensitive operations
        privacy_logger = logging.getLogger('privacy')
        privacy_logger.info(message)
        self._log_to_database('INFO', message, user_action=operation)
    
    def log_performance_metric(self, operation: str, duration: float, 
                             memory_usage: Optional[float] = None):
        """Log performance metrics"""
        message = f"Performance: {operation} took {duration:.2f}s"
        if memory_usage:
            message += f", memory: {memory_usage:.2f}MB"
        
        self.debug(message, user_action='performance_monitoring')

class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: SafeDataLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        self.start_memory = None
    
    def __enter__(self):
        import time
        import psutil
        
        self.start_time = time.time()
        try:
            process = psutil.Process()
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB
        except:
            self.start_memory = None
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        import psutil
        
        duration = time.time() - self.start_time
        
        memory_usage = None
        if self.start_memory:
            try:
                process = psutil.Process()
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_usage = current_memory - self.start_memory
            except:
                pass
        
        self.logger.log_performance_metric(self.operation, duration, memory_usage)

class SensitiveDataFilter(logging.Filter):
    """Filter to prevent logging of sensitive data"""
    
    SENSITIVE_PATTERNS = [
        # Common patterns that might contain sensitive data
        r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
        r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
        r'\b\d{10,}\b',  # Long numbers that might be IDs
    ]
    
    def __init__(self):
        super().__init__()
        import re
        self.patterns = [re.compile(pattern) for pattern in self.SENSITIVE_PATTERNS]
    
    def filter(self, record):
        """Filter out records containing sensitive data patterns"""
        message = record.getMessage()
        
        for pattern in self.patterns:
            if pattern.search(message):
                # Replace sensitive data with placeholder
                message = pattern.sub('[REDACTED]', message)
                record.msg = message
                record.args = ()
        
        return True

def setup_audit_logging(audit_log_file: Optional[str] = None) -> logging.Logger:
    """
    Setup dedicated audit logging for compliance requirements
    
    Args:
        audit_log_file: Path to audit log file
        
    Returns:
        Configured audit logger
    """
    
    # Create audit logs directory
    if audit_log_file is None:
        audit_dir = Path.home() / ".safedata" / "audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_log_file = audit_dir / "safedata_audit.log"
    else:
        audit_log_file = Path(audit_log_file)
        audit_log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create audit logger
    audit_logger = logging.getLogger('audit')
    audit_logger.setLevel(logging.INFO)
    
    # Create audit formatter
    audit_formatter = logging.Formatter(
        fmt='%(asctime)s - AUDIT - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create audit file handler (no rotation for compliance)
    audit_handler = logging.FileHandler(
        filename=str(audit_log_file),
        encoding='utf-8'
    )
    audit_handler.setLevel(logging.INFO)
    audit_handler.setFormatter(audit_formatter)
    
    # Add sensitive data filter
    audit_handler.addFilter(SensitiveDataFilter())
    
    audit_logger.addHandler(audit_handler)
    audit_logger.propagate = False  # Don't propagate to root logger
    
    audit_logger.info("Audit logging initialized")
    
    return audit_logger

def get_logger(name: str, session_id: Optional[str] = None) -> SafeDataLogger:
    """
    Get a SafeData logger instance
    
    Args:
        name: Logger name
        session_id: Optional session ID for context
        
    Returns:
        SafeDataLogger instance
    """
    return SafeDataLogger(name, session_id)

def log_system_info():
    """Log system information for debugging purposes"""
    
    logger = logging.getLogger(__name__)
    
    try:
        import platform
        import sys
        import psutil
        
        logger.info("=== System Information ===")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Python: {sys.version}")
        logger.info(f"Architecture: {platform.architecture()}")
        logger.info(f"Processor: {platform.processor()}")
        
        # Memory info
        memory = psutil.virtual_memory()
        logger.info(f"Total Memory: {memory.total / 1024 / 1024 / 1024:.1f} GB")
        logger.info(f"Available Memory: {memory.available / 1024 / 1024 / 1024:.1f} GB")
        
        # Disk info
        disk = psutil.disk_usage('/')
        logger.info(f"Disk Total: {disk.total / 1024 / 1024 / 1024:.1f} GB")
        logger.info(f"Disk Free: {disk.free / 1024 / 1024 / 1024:.1f} GB")
        
        logger.info("=== End System Information ===")
        
    except Exception as e:
        logger.warning(f"Could not log system information: {str(e)}")

def configure_third_party_loggers():
    """Configure logging levels for third-party libraries"""
    
    # Reduce logging level for noisy third-party libraries
    noisy_loggers = [
        'matplotlib',
        'PIL',
        'urllib3',
        'requests',
        'tensorflow',
        'sklearn'
    ]
    
    for logger_name in noisy_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.WARNING)

def setup_development_logging():
    """Setup logging configuration for development environment"""
    
    setup_logging(
        log_level="DEBUG",
        enable_console=True,
        enable_file=True
    )
    
    # Log system info in development
    log_system_info()
    
    # Configure third-party loggers
    configure_third_party_loggers()

def setup_production_logging():
    """Setup logging configuration for production environment"""
    
    setup_logging(
        log_level="INFO",
        enable_console=False,
        enable_file=True,
        max_file_size=50 * 1024 * 1024,  # 50MB
        backup_count=10
    )
    
    # Setup audit logging for production
    setup_audit_logging()
    
    # Configure third-party loggers
    configure_third_party_loggers()

# Module-level initialization
def init_logging():
    """Initialize logging based on environment"""
    
    # Check if we're in development mode
    if os.getenv('SAFEDATA_ENV', 'production').lower() == 'development':
        setup_development_logging()
    else:
        setup_production_logging()

# Auto-initialize if imported
if __name__ != '__main__':
    init_logging()
