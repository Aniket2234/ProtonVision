"""
Main Window GUI for SafeData Pipeline
Professional interface with dashboard and navigation
"""

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, filedialog
import threading
import logging
from pathlib import Path

from gui.dashboard import DashboardFrame
from gui.config_panel import ConfigurationPanel
from gui.widgets import StatusBar, ProgressDialog
from modules.data_loader import DataLoader
from modules.risk_assessment import RiskAssessment
from modules.privacy_enhancement import PrivacyEnhancement
from modules.utility_measurement import UtilityMeasurement
from reports.report_generator import ReportGenerator
from assets.styles import APP_STYLES

class SafeDataMainWindow:
    """Main application window with tabbed interface"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.setup_window()
        self.setup_modules()
        self.setup_gui()
        self.current_dataset = None
        self.processing_thread = None
        
    def setup_window(self):
        """Initialize main window"""
        self.root = ctk.CTk()
        self.root.title("SafeData Pipeline - Privacy-Preserving Data Analysis")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def setup_modules(self):
        """Initialize core modules"""
        self.data_loader = DataLoader()
        self.risk_assessment = RiskAssessment()
        self.privacy_enhancement = PrivacyEnhancement()
        self.utility_measurement = UtilityMeasurement()
        self.report_generator = ReportGenerator()
        
    def setup_gui(self):
        """Setup the main GUI components"""
        self.create_header()
        self.create_main_content()
        self.create_status_bar()
        
    def create_header(self):
        """Create application header with navigation"""
        header_frame = ctk.CTkFrame(self.root, height=80)
        header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        header_frame.grid_columnconfigure(1, weight=1)
        
        # Logo and title
        title_label = ctk.CTkLabel(
            header_frame,
            text="SafeData Pipeline",
            font=ctk.CTkFont(size=24, weight="bold"),
            text_color=APP_STYLES["primary_color"]
        )
        title_label.grid(row=0, column=0, padx=20, pady=20, sticky="w")
        
        # Navigation buttons
        nav_frame = ctk.CTkFrame(header_frame)
        nav_frame.grid(row=0, column=1, padx=20, pady=10, sticky="e")
        
        self.nav_buttons = {}
        nav_items = [
            ("Dashboard", self.show_dashboard),
            ("Data Upload", self.show_data_upload),
            ("Risk Assessment", self.show_risk_assessment),
            ("Privacy Enhancement", self.show_privacy_enhancement),
            ("Utility Analysis", self.show_utility_analysis),
            ("Reports", self.show_reports)
        ]
        
        for i, (text, command) in enumerate(nav_items):
            btn = ctk.CTkButton(
                nav_frame,
                text=text,
                command=command,
                width=120,
                height=35,
                font=ctk.CTkFont(size=12)
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.nav_buttons[text] = btn
    
    def create_main_content(self):
        """Create main content area with tabbed interface"""
        self.content_frame = ctk.CTkFrame(self.root)
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.content_frame.grid_rowconfigure(0, weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)
        
        # Initialize different views
        self.dashboard = DashboardFrame(self.content_frame, self)
        self.config_panel = ConfigurationPanel(self.content_frame, self)
        
        # Show dashboard by default
        self.current_view = "dashboard"
        self.dashboard.show()
        
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = StatusBar(self.root)
        self.status_bar.grid(row=2, column=0, sticky="ew", padx=10, pady=(5, 10))
        
    def update_navigation(self, active_view):
        """Update navigation button states"""
        for name, button in self.nav_buttons.items():
            if name.lower().replace(" ", "_") == active_view:
                button.configure(fg_color=APP_STYLES["accent_color"])
            else:
                button.configure(fg_color=APP_STYLES["button_color"])
    
    def show_dashboard(self):
        """Show dashboard view"""
        self.hide_all_views()
        self.dashboard.show()
        self.current_view = "dashboard"
        self.update_navigation("dashboard")
        
    def show_data_upload(self):
        """Show data upload dialog"""
        self.upload_data()
        
    def show_risk_assessment(self):
        """Show risk assessment view"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
        self.hide_all_views()
        self.config_panel.show_risk_assessment()
        self.current_view = "risk_assessment"
        self.update_navigation("risk_assessment")
        
    def show_privacy_enhancement(self):
        """Show privacy enhancement view"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
        self.hide_all_views()
        self.config_panel.show_privacy_enhancement()
        self.current_view = "privacy_enhancement"
        self.update_navigation("privacy_enhancement")
        
    def show_utility_analysis(self):
        """Show utility analysis view"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
        self.hide_all_views()
        self.config_panel.show_utility_analysis()
        self.current_view = "utility_analysis"
        self.update_navigation("utility_analysis")
        
    def show_reports(self):
        """Show reports view"""
        self.hide_all_views()
        self.config_panel.show_reports()
        self.current_view = "reports"
        self.update_navigation("reports")
        
    def hide_all_views(self):
        """Hide all content views"""
        self.dashboard.hide()
        self.config_panel.hide()
        
    def upload_data(self):
        """Handle data upload"""
        filetypes = [
            ("CSV files", "*.csv"),
            ("Excel files", "*.xlsx *.xls"),
            ("JSON files", "*.json"),
            ("All files", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Select Dataset",
            filetypes=filetypes
        )
        
        if filename:
            try:
                # Show progress dialog
                progress = ProgressDialog(self.root, "Loading Data", "Loading dataset...")
                progress.show()
                
                # Load data in background thread
                def load_data():
                    try:
                        self.current_dataset = self.data_loader.load_file(filename)
                        self.root.after(0, lambda: self.on_data_loaded(progress))
                    except Exception as e:
                        self.root.after(0, lambda: self.on_data_load_error(progress, str(e)))
                
                threading.Thread(target=load_data, daemon=True).start()
                
            except Exception as e:
                messagebox.showerror("Load Error", f"Failed to load dataset: {str(e)}")
                
    def on_data_loaded(self, progress):
        """Handle successful data loading"""
        progress.close()
        self.status_bar.set_status(f"Dataset loaded: {len(self.current_dataset)} records")
        self.dashboard.update_data_info(self.current_dataset)
        messagebox.showinfo("Success", "Dataset loaded successfully!")
        
    def on_data_load_error(self, progress, error_msg):
        """Handle data loading error"""
        progress.close()
        messagebox.showerror("Load Error", f"Failed to load dataset: {error_msg}")
        
    def run_risk_assessment(self, config):
        """Run risk assessment with given configuration"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
            
        def assess_risk():
            try:
                results = self.risk_assessment.assess_dataset(self.current_dataset, config)
                self.root.after(0, lambda: self.on_risk_assessment_complete(results))
            except Exception as e:
                self.root.after(0, lambda: self.on_processing_error("Risk Assessment", str(e)))
        
        self.processing_thread = threading.Thread(target=assess_risk, daemon=True)
        self.processing_thread.start()
        
    def run_privacy_enhancement(self, config):
        """Run privacy enhancement with given configuration"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
            
        def enhance_privacy():
            try:
                results = self.privacy_enhancement.enhance_dataset(self.current_dataset, config)
                self.root.after(0, lambda: self.on_privacy_enhancement_complete(results))
            except Exception as e:
                self.root.after(0, lambda: self.on_processing_error("Privacy Enhancement", str(e)))
        
        self.processing_thread = threading.Thread(target=enhance_privacy, daemon=True)
        self.processing_thread.start()
        
    def run_utility_analysis(self, config):
        """Run utility analysis with given configuration"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
            
        def analyze_utility():
            try:
                results = self.utility_measurement.analyze_dataset(self.current_dataset, config)
                self.root.after(0, lambda: self.on_utility_analysis_complete(results))
            except Exception as e:
                self.root.after(0, lambda: self.on_processing_error("Utility Analysis", str(e)))
        
        self.processing_thread = threading.Thread(target=analyze_utility, daemon=True)
        self.processing_thread.start()
        
    def on_risk_assessment_complete(self, results):
        """Handle completed risk assessment"""
        self.dashboard.update_risk_metrics(results)
        self.status_bar.set_status("Risk assessment completed")
        messagebox.showinfo("Complete", "Risk assessment completed successfully!")
        
    def on_privacy_enhancement_complete(self, results):
        """Handle completed privacy enhancement"""
        self.dashboard.update_privacy_metrics(results)
        self.status_bar.set_status("Privacy enhancement completed")
        messagebox.showinfo("Complete", "Privacy enhancement completed successfully!")
        
    def on_utility_analysis_complete(self, results):
        """Handle completed utility analysis"""
        self.dashboard.update_utility_metrics(results)
        self.status_bar.set_status("Utility analysis completed")
        messagebox.showinfo("Complete", "Utility analysis completed successfully!")
        
    def on_processing_error(self, operation, error_msg):
        """Handle processing errors"""
        self.status_bar.set_status(f"{operation} failed")
        messagebox.showerror("Error", f"{operation} failed: {error_msg}")
        
    def generate_report(self, report_type):
        """Generate and save report"""
        if not self.current_dataset:
            messagebox.showwarning("No Data", "Please upload a dataset first.")
            return
            
        filename = filedialog.asksaveasfilename(
            title="Save Report",
            defaultextension=".pdf",
            filetypes=[("PDF files", "*.pdf"), ("HTML files", "*.html")]
        )
        
        if filename:
            try:
                self.report_generator.generate_report(
                    self.current_dataset,
                    report_type,
                    filename
                )
                messagebox.showinfo("Success", f"Report saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to generate report: {str(e)}")
                
    def run(self):
        """Start the application"""
        self.logger.info("Starting SafeData Pipeline GUI")
        self.root.mainloop()
