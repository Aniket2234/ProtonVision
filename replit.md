# SafeData Pipeline - Privacy-Preserving Data Analysis Tool

## Overview

SafeData Pipeline is a comprehensive desktop application built with Python and Tkinter that enables privacy-preserving data analysis. The application provides a professional interface for assessing privacy risks, applying enhancement techniques, and measuring data utility while maintaining statistical value. It implements multiple privacy preservation methods including Statistical Disclosure Control, Differential Privacy, and Synthetic Data Generation.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **GUI Framework**: CustomTkinter (CTk) for modern, dark-themed interface
- **Layout System**: Grid-based layout with tabbed navigation
- **Visualization**: Matplotlib integration for real-time charts and graphs
- **Component Structure**: Modular widget system with reusable components
- **Styling**: Centralized styling system with consistent color palette and themes

### Backend Architecture
- **Core Language**: Python 3.x
- **Architecture Pattern**: Modular design with separated concerns
- **Data Processing**: Pandas for data manipulation and analysis
- **Privacy Techniques**: Custom implementations of privacy-preserving algorithms
- **Machine Learning**: Scikit-learn for utility measurement and synthetic data generation
- **Deep Learning**: TensorFlow/Keras for advanced synthetic data generation (optional)

### Data Storage Solutions
- **Primary Database**: SQLite for local application state and audit logs
- **File Support**: CSV, Excel, JSON, Parquet, TSV file formats
- **Data Persistence**: Pickle serialization for complex objects
- **Backup Strategy**: Automated database backups with configurable intervals

## Key Components

### GUI Components
1. **Main Window** (`main_window.py`): Central application interface with tabbed navigation
2. **Dashboard** (`dashboard.py`): Real-time metrics visualization and monitoring
3. **Configuration Panel** (`config_panel.py`): Parameter settings for privacy techniques
4. **Custom Widgets** (`widgets.py`): Reusable UI components (MetricsCard, ProgressDialog, etc.)

### Core Processing Modules
1. **Data Loader** (`data_loader.py`): Multi-format data ingestion and validation
2. **Risk Assessment** (`risk_assessment.py`): Privacy risk evaluation through attack simulation
3. **Privacy Enhancement** (`privacy_enhancement.py`): Coordinated application of privacy techniques
4. **Utility Measurement** (`utility_measurement.py`): Quantification of analytical value preservation

### Privacy Techniques
1. **Statistical Disclosure Control** (`statistical_disclosure.py`): Suppression, generalization, perturbation
2. **Differential Privacy** (`differential_privacy.py`): Laplace, Gaussian, and exponential mechanisms
3. **Synthetic Data Generation** (`synthetic_data.py`): GANs, VAEs, Bayesian Networks, Copula methods

### Utility and Support
1. **Attack Simulation** (`attack_simulation.py`): Linkage attack simulations for risk assessment
2. **Privacy Metrics** (`privacy_metrics.py`): K-anonymity, L-diversity, T-closeness calculations
3. **Report Generator** (`report_generator.py`): Multi-format report generation (PDF, HTML)
4. **Database Manager** (`database.py`): SQLite operations for state management

## Data Flow

1. **Data Ingestion**: Files loaded through DataLoader with format detection and validation
2. **Risk Assessment**: Original dataset analyzed for privacy vulnerabilities using simulated attacks
3. **Privacy Enhancement**: Selected techniques applied based on configuration parameters
4. **Utility Measurement**: Enhanced dataset compared against original for utility preservation
5. **Reporting**: Comprehensive reports generated with visualizations and recommendations
6. **Audit Logging**: All operations logged to SQLite database for traceability

## External Dependencies

### Core Dependencies
- **customtkinter**: Modern GUI framework for professional interface
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning utilities
- **scipy**: Scientific computing

### Optional Dependencies
- **tensorflow**: Deep learning for advanced synthetic data generation
- **reportlab**: PDF report generation
- **jinja2**: HTML template rendering
- **openpyxl**: Excel file support

### File Format Support
- **CSV/TSV**: Standard delimited files
- **Excel**: .xlsx and .xls formats
- **JSON**: Structured data format
- **Parquet**: Columnar storage format

## Deployment Strategy

### Application Structure
- **Standalone Desktop Application**: Self-contained Python application
- **Local Data Storage**: All data processing and storage happens locally
- **Configuration Management**: User-specific settings stored in home directory
- **Cross-Platform**: Compatible with Windows, macOS, and Linux

### Directory Structure
- **Configuration**: `~/.safedata/config/` - Application settings
- **Data**: `~/.safedata/data/` - Database and cached data
- **Logs**: `~/.safedata/logs/` - Application logs with rotation
- **Reports**: `~/.safedata/reports/` - Generated analysis reports
- **Cache**: `~/.safedata/cache/` - Temporary processing files

### Security Considerations
- **Local Processing**: No data transmitted to external services
- **Privacy by Design**: Built-in privacy protection mechanisms
- **Audit Trail**: Complete logging of all privacy operations
- **Data Isolation**: User data contained within local application directory

The application follows a modular, extensible design that allows for easy addition of new privacy techniques and utility metrics while maintaining a professional, user-friendly interface.