"""
SafeData Pipeline - Main Application Entry Point
Professional Privacy-Preserving Data Analysis Tool
"""

import tkinter as tk
import customtkinter as ctk
import sys
import os
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gui.main_window import SafeDataMainWindow
from utils.logger import setup_logging
from utils.database import DatabaseManager
from config.settings import get_app_info

class SafeDataPipeline:
    """Main application class for SafeData Pipeline"""
    
    def __init__(self):
        self.setup_application()
        self.db_manager = DatabaseManager()
        self.main_window = None
    
    def setup_application(self):
        """Setup application configuration and logging"""
        # Setup logging
        setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Set CustomTkinter appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        self.logger.info("SafeData Pipeline application initialized")
    
    def run(self):
        """Run the main application"""
        try:
            # Initialize database
            self.db_manager.initialize_database()
            
            # Create and run main window
            self.main_window = SafeDataMainWindow()
            self.main_window.run()
            
        except Exception as e:
            self.logger.error(f"Application error: {str(e)}")
            raise
        finally:
            if self.db_manager:
                self.db_manager.close()

def main():
    """Application entry point"""
    try:
        app = SafeDataPipeline()
        app.run()
    except KeyboardInterrupt:
        print("\nApplication terminated by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
