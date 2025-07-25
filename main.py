"""
SafeData Pipeline - Main Application Entry Point
Privacy-Preserving Data Analysis Tool for Replit
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the web application
from app import app, safedata_app

def main():
    """Application entry point for Replit"""
    try:
        print("Starting SafeData Pipeline Web Application...")
        port = int(os.environ.get('PORT', 5000))
        app.run(host='0.0.0.0', port=port, debug=False)
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
