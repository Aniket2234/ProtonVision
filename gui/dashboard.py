"""
Dashboard GUI component for SafeData Pipeline
Real-time visualization and monitoring interface
"""

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
from datetime import datetime
import logging

from gui.widgets import MetricsCard, PrivacyHeatmap
from assets.styles import APP_STYLES

class DashboardFrame:
    """Main dashboard with real-time metrics and visualizations"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.setup_dashboard()
        self.is_visible = False
        
    def setup_dashboard(self):
        """Setup dashboard layout and components"""
        self.frame = ctk.CTkFrame(self.parent)
        self.frame.grid_rowconfigure(1, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        self.create_overview_section()
        self.create_metrics_section()
        self.create_visualizations_section()
        
    def create_overview_section(self):
        """Create overview section with key metrics"""
        overview_frame = ctk.CTkFrame(self.frame)
        overview_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        overview_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            overview_frame,
            text="Privacy-Utility Dashboard",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        title_label.grid(row=0, column=0, columnspan=4, pady=(10, 20))
        
        # Metrics cards
        self.data_info_card = MetricsCard(
            overview_frame,
            "Dataset Info",
            "No data loaded",
            APP_STYLES["info_color"]
        )
        self.data_info_card.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        self.risk_card = MetricsCard(
            overview_frame,
            "Privacy Risk",
            "Not assessed",
            APP_STYLES["warning_color"]
        )
        self.risk_card.grid(row=1, column=1, padx=10, pady=10, sticky="ew")
        
        self.utility_card = MetricsCard(
            overview_frame,
            "Data Utility",
            "Not measured",
            APP_STYLES["success_color"]
        )
        self.utility_card.grid(row=1, column=2, padx=10, pady=10, sticky="ew")
        
        self.compliance_card = MetricsCard(
            overview_frame,
            "DPDP Compliance",
            "Unknown",
            APP_STYLES["accent_color"]
        )
        self.compliance_card.grid(row=1, column=3, padx=10, pady=10, sticky="ew")
        
    def create_metrics_section(self):
        """Create detailed metrics section"""
        metrics_frame = ctk.CTkFrame(self.frame)
        metrics_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        metrics_frame.grid_rowconfigure(0, weight=1)
        metrics_frame.grid_columnconfigure((0, 1), weight=1)
        
        # Risk Assessment Panel
        self.create_risk_panel(metrics_frame)
        
        # Privacy Enhancement Panel
        self.create_privacy_panel(metrics_frame)
        
    def create_risk_panel(self, parent):
        """Create risk assessment visualization panel"""
        risk_frame = ctk.CTkFrame(parent)
        risk_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        risk_frame.grid_rowconfigure(1, weight=1)
        risk_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            risk_frame,
            text="Risk Assessment",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=10)
        
        # Risk visualization canvas
        self.risk_canvas_frame = ctk.CTkFrame(risk_frame)
        self.risk_canvas_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
        # Initialize risk plot
        self.setup_risk_plot()
        
    def create_privacy_panel(self, parent):
        """Create privacy enhancement visualization panel"""
        privacy_frame = ctk.CTkFrame(parent)
        privacy_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        privacy_frame.grid_rowconfigure(1, weight=1)
        privacy_frame.grid_columnconfigure(0, weight=1)
        
        # Title
        title_label = ctk.CTkLabel(
            privacy_frame,
            text="Privacy Enhancement",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        title_label.grid(row=0, column=0, pady=10)
        
        # Privacy heatmap
        self.privacy_heatmap = PrivacyHeatmap(privacy_frame)
        self.privacy_heatmap.grid(row=1, column=0, sticky="nsew", padx=10, pady=10)
        
    def create_visualizations_section(self):
        """Create additional visualizations section"""
        viz_frame = ctk.CTkFrame(self.frame)
        viz_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        viz_frame.grid_columnconfigure((0, 1, 2), weight=1)
        
        # Utility trend chart
        self.create_utility_trend(viz_frame)
        
        # Privacy budget tracker
        self.create_privacy_budget(viz_frame)
        
        # Processing status
        self.create_processing_status(viz_frame)
        
    def setup_risk_plot(self):
        """Setup risk assessment plot"""
        plt.style.use('dark_background')
        self.risk_fig, self.risk_ax = plt.subplots(figsize=(6, 4))
        self.risk_fig.patch.set_facecolor('#2b2b2b')
        
        # Initialize empty plot
        self.risk_ax.set_title("Individual Risk Distribution", color='white')
        self.risk_ax.set_xlabel("Risk Score", color='white')
        self.risk_ax.set_ylabel("Frequency", color='white')
        self.risk_ax.tick_params(colors='white')
        
        # Embed plot in canvas
        self.risk_canvas = FigureCanvasTkAgg(self.risk_fig, self.risk_canvas_frame)
        self.risk_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def create_utility_trend(self, parent):
        """Create utility trend visualization"""
        utility_frame = ctk.CTkFrame(parent)
        utility_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            utility_frame,
            text="Utility Trend",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Utility metrics display
        self.utility_metrics = ctk.CTkTextbox(utility_frame, height=100)
        self.utility_metrics.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.utility_metrics.insert("0.0", "No utility analysis performed yet.")
        
    def create_privacy_budget(self, parent):
        """Create privacy budget tracker"""
        budget_frame = ctk.CTkFrame(parent)
        budget_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            budget_frame,
            text="Privacy Budget",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Budget progress bar
        self.budget_progress = ctk.CTkProgressBar(budget_frame)
        self.budget_progress.pack(padx=20, pady=10, fill=tk.X)
        self.budget_progress.set(0.0)
        
        self.budget_label = ctk.CTkLabel(budget_frame, text="ε = 0.0 / 1.0")
        self.budget_label.pack(pady=5)
        
    def create_processing_status(self, parent):
        """Create processing status panel"""
        status_frame = ctk.CTkFrame(parent)
        status_frame.grid(row=0, column=2, sticky="nsew", padx=5, pady=5)
        
        title_label = ctk.CTkLabel(
            status_frame,
            text="Processing Status",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        title_label.pack(pady=10)
        
        # Status indicators
        self.status_indicators = {}
        operations = ["Data Loading", "Risk Assessment", "Privacy Enhancement", "Utility Analysis"]
        
        for op in operations:
            indicator = ctk.CTkLabel(
                status_frame,
                text=f"● {op}: Ready",
                text_color="gray"
            )
            indicator.pack(pady=2, anchor="w", padx=20)
            self.status_indicators[op] = indicator
    
    def update_data_info(self, dataset):
        """Update dataset information display"""
        if dataset is not None:
            rows, cols = dataset.shape
            info_text = f"{rows:,} rows\n{cols} columns"
            self.data_info_card.update_value(info_text)
            
            # Update status
            self.status_indicators["Data Loading"].configure(
                text="● Data Loading: Complete",
                text_color=APP_STYLES["success_color"]
            )
            
    def update_risk_metrics(self, risk_results):
        """Update risk assessment metrics display"""
        if risk_results:
            # Update risk card
            global_risk = risk_results.get('global_risk', 0)
            risk_level = "Low" if global_risk < 0.3 else "Medium" if global_risk < 0.7 else "High"
            self.risk_card.update_value(f"{risk_level}\n{global_risk:.3f}")
            
            # Update risk plot
            if 'individual_risks' in risk_results:
                self.risk_ax.clear()
                self.risk_ax.hist(
                    risk_results['individual_risks'],
                    bins=30,
                    alpha=0.7,
                    color=APP_STYLES["primary_color"]
                )
                self.risk_ax.set_title("Individual Risk Distribution", color='white')
                self.risk_ax.set_xlabel("Risk Score", color='white')
                self.risk_ax.set_ylabel("Frequency", color='white')
                self.risk_canvas.draw()
            
            # Update status
            self.status_indicators["Risk Assessment"].configure(
                text="● Risk Assessment: Complete",
                text_color=APP_STYLES["success_color"]
            )
            
    def update_privacy_metrics(self, privacy_results):
        """Update privacy enhancement metrics display"""
        if privacy_results:
            # Update privacy heatmap
            self.privacy_heatmap.update_data(privacy_results.get('privacy_map', {}))
            
            # Update budget tracker
            budget_used = privacy_results.get('budget_used', 0)
            self.budget_progress.set(budget_used)
            self.budget_label.configure(text=f"ε = {budget_used:.3f} / 1.0")
            
            # Update status
            self.status_indicators["Privacy Enhancement"].configure(
                text="● Privacy Enhancement: Complete",
                text_color=APP_STYLES["success_color"]
            )
            
    def update_utility_metrics(self, utility_results):
        """Update utility analysis metrics display"""
        if utility_results:
            # Update utility card
            overall_utility = utility_results.get('overall_utility', 0)
            utility_score = f"{overall_utility:.2%}"
            self.utility_card.update_value(f"Preserved\n{utility_score}")
            
            # Update utility trend
            self.utility_metrics.delete("0.0", tk.END)
            metrics_text = f"""Statistical Utility:
MAE: {utility_results.get('mae', 0):.4f}
RMSE: {utility_results.get('rmse', 0):.4f}
Correlation: {utility_results.get('correlation', 0):.4f}

Analytical Utility:
Query Accuracy: {utility_results.get('query_accuracy', 0):.2%}
Model Performance: {utility_results.get('model_performance', 0):.2%}"""
            
            self.utility_metrics.insert("0.0", metrics_text)
            
            # Update status
            self.status_indicators["Utility Analysis"].configure(
                text="● Utility Analysis: Complete",
                text_color=APP_STYLES["success_color"]
            )
            
    def show(self):
        """Show the dashboard"""
        if not self.is_visible:
            self.frame.grid(row=0, column=0, sticky="nsew")
            self.is_visible = True
            
    def hide(self):
        """Hide the dashboard"""
        if self.is_visible:
            self.frame.grid_forget()
            self.is_visible = False
