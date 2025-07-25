"""
Configuration Panel GUI for SafeData Pipeline
Parameter settings and technique selection interface
"""

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox, ttk
import json
import logging

from assets.styles import APP_STYLES

class ConfigurationPanel:
    """Configuration panel for privacy enhancement parameters"""
    
    def __init__(self, parent, main_window):
        self.parent = parent
        self.main_window = main_window
        self.logger = logging.getLogger(__name__)
        self.current_config = {}
        self.setup_panel()
        self.is_visible = False
        
    def setup_panel(self):
        """Setup configuration panel layout"""
        self.frame = ctk.CTkFrame(self.parent)
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)
        
        # Create notebook for different configuration sections
        self.notebook = ctk.CTkTabview(self.frame)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Add tabs
        self.notebook.add("Risk Assessment")
        self.notebook.add("Privacy Enhancement")
        self.notebook.add("Utility Analysis")
        self.notebook.add("Reports")
        
        self.setup_risk_assessment_tab()
        self.setup_privacy_enhancement_tab()
        self.setup_utility_analysis_tab()
        self.setup_reports_tab()
        
    def setup_risk_assessment_tab(self):
        """Setup risk assessment configuration tab"""
        tab = self.notebook.tab("Risk Assessment")
        
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Quasi-identifier selection
        qi_frame = ctk.CTkFrame(scroll_frame)
        qi_frame.pack(fill=tk.X, padx=5, pady=10)
        
        qi_label = ctk.CTkLabel(
            qi_frame,
            text="Quasi-Identifier Configuration",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        qi_label.pack(pady=10)
        
        # Auto-detect QI checkbox
        self.auto_detect_qi = ctk.CTkCheckBox(
            qi_frame,
            text="Auto-detect quasi-identifiers",
            command=self.toggle_qi_selection
        )
        self.auto_detect_qi.pack(pady=5)
        self.auto_detect_qi.select()
        
        # Manual QI selection
        self.qi_listbox_frame = ctk.CTkFrame(qi_frame)
        self.qi_listbox_frame.pack(fill=tk.X, padx=10, pady=5)
        
        qi_list_label = ctk.CTkLabel(self.qi_listbox_frame, text="Select Quasi-Identifiers:")
        qi_list_label.pack(anchor="w", padx=5, pady=5)
        
        # QI selection will be populated when data is loaded
        self.qi_vars = {}
        
        # Attack simulation parameters
        attack_frame = ctk.CTkFrame(scroll_frame)
        attack_frame.pack(fill=tk.X, padx=5, pady=10)
        
        attack_label = ctk.CTkLabel(
            attack_frame,
            text="Attack Simulation Parameters",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        attack_label.pack(pady=10)
        
        # Attack types
        attack_types = ["Record Linkage", "Attribute Linkage", "Membership Inference", "Homogeneity Attack"]
        self.attack_vars = {}
        
        for attack_type in attack_types:
            var = ctk.CTkCheckBox(attack_frame, text=attack_type)
            var.pack(anchor="w", padx=20, pady=2)
            var.select()
            self.attack_vars[attack_type] = var
            
        # Risk thresholds
        threshold_frame = ctk.CTkFrame(scroll_frame)
        threshold_frame.pack(fill=tk.X, padx=5, pady=10)
        
        threshold_label = ctk.CTkLabel(
            threshold_frame,
            text="Risk Thresholds",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        threshold_label.pack(pady=10)
        
        # Individual risk threshold
        ctk.CTkLabel(threshold_frame, text="Individual Risk Threshold:").pack(anchor="w", padx=20)
        self.individual_threshold = ctk.CTkSlider(
            threshold_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100
        )
        self.individual_threshold.pack(fill=tk.X, padx=20, pady=5)
        self.individual_threshold.set(0.1)
        
        self.individual_threshold_label = ctk.CTkLabel(threshold_frame, text="0.10")
        self.individual_threshold_label.pack(padx=20)
        self.individual_threshold.configure(command=self.update_individual_threshold)
        
        # Global risk threshold
        ctk.CTkLabel(threshold_frame, text="Global Risk Threshold:").pack(anchor="w", padx=20, pady=(10, 0))
        self.global_threshold = ctk.CTkSlider(
            threshold_frame,
            from_=0.0,
            to=1.0,
            number_of_steps=100
        )
        self.global_threshold.pack(fill=tk.X, padx=20, pady=5)
        self.global_threshold.set(0.05)
        
        self.global_threshold_label = ctk.CTkLabel(threshold_frame, text="0.05")
        self.global_threshold_label.pack(padx=20)
        self.global_threshold.configure(command=self.update_global_threshold)
        
        # Run button
        run_button = ctk.CTkButton(
            scroll_frame,
            text="Run Risk Assessment",
            command=self.run_risk_assessment,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        run_button.pack(pady=20)
        
    def setup_privacy_enhancement_tab(self):
        """Setup privacy enhancement configuration tab"""
        tab = self.notebook.tab("Privacy Enhancement")
        
        # Create scrollable frame
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Technique selection
        technique_frame = ctk.CTkFrame(scroll_frame)
        technique_frame.pack(fill=tk.X, padx=5, pady=10)
        
        technique_label = ctk.CTkLabel(
            technique_frame,
            text="Privacy Enhancement Techniques",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        technique_label.pack(pady=10)
        
        # Technique tabs
        self.technique_notebook = ctk.CTkTabview(technique_frame)
        self.technique_notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.technique_notebook.add("Statistical Disclosure Control")
        self.technique_notebook.add("Differential Privacy")
        self.technique_notebook.add("Synthetic Data Generation")
        
        self.setup_sdc_config()
        self.setup_dp_config()
        self.setup_sdg_config()
        
        # Run button
        run_button = ctk.CTkButton(
            scroll_frame,
            text="Run Privacy Enhancement",
            command=self.run_privacy_enhancement,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        run_button.pack(pady=20)
        
    def setup_sdc_config(self):
        """Setup Statistical Disclosure Control configuration"""
        tab = self.technique_notebook.tab("Statistical Disclosure Control")
        
        # SDC techniques
        sdc_frame = ctk.CTkFrame(tab)
        sdc_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Suppression
        suppression_frame = ctk.CTkFrame(sdc_frame)
        suppression_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_suppression = ctk.CTkCheckBox(
            suppression_frame,
            text="Suppression",
            command=self.toggle_suppression_config
        )
        self.use_suppression.pack(anchor="w", padx=10, pady=5)
        
        self.suppression_config = ctk.CTkFrame(suppression_frame)
        
        ctk.CTkLabel(self.suppression_config, text="Suppression Threshold:").pack(anchor="w")
        self.suppression_threshold = ctk.CTkEntry(self.suppression_config, placeholder_text="5")
        self.suppression_threshold.pack(fill=tk.X, padx=5, pady=2)
        
        # Generalization
        generalization_frame = ctk.CTkFrame(sdc_frame)
        generalization_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_generalization = ctk.CTkCheckBox(
            generalization_frame,
            text="Generalization",
            command=self.toggle_generalization_config
        )
        self.use_generalization.pack(anchor="w", padx=10, pady=5)
        
        self.generalization_config = ctk.CTkFrame(generalization_frame)
        
        ctk.CTkLabel(self.generalization_config, text="Generalization Levels:").pack(anchor="w")
        self.generalization_levels = ctk.CTkEntry(self.generalization_config, placeholder_text="2")
        self.generalization_levels.pack(fill=tk.X, padx=5, pady=2)
        
        # Perturbation
        perturbation_frame = ctk.CTkFrame(sdc_frame)
        perturbation_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.use_perturbation = ctk.CTkCheckBox(
            perturbation_frame,
            text="Perturbation",
            command=self.toggle_perturbation_config
        )
        self.use_perturbation.pack(anchor="w", padx=10, pady=5)
        
        self.perturbation_config = ctk.CTkFrame(perturbation_frame)
        
        ctk.CTkLabel(self.perturbation_config, text="Noise Level:").pack(anchor="w")
        self.noise_level = ctk.CTkSlider(self.perturbation_config, from_=0.0, to=1.0)
        self.noise_level.pack(fill=tk.X, padx=5, pady=2)
        self.noise_level.set(0.1)
        
    def setup_dp_config(self):
        """Setup Differential Privacy configuration"""
        tab = self.technique_notebook.tab("Differential Privacy")
        
        dp_frame = ctk.CTkFrame(tab)
        dp_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Privacy budget
        budget_frame = ctk.CTkFrame(dp_frame)
        budget_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(budget_frame, text="Privacy Budget (ε):").pack(anchor="w", padx=10, pady=5)
        self.epsilon = ctk.CTkEntry(budget_frame, placeholder_text="1.0")
        self.epsilon.pack(fill=tk.X, padx=10, pady=5)
        
        # Delta parameter
        ctk.CTkLabel(budget_frame, text="Delta (δ):").pack(anchor="w", padx=10, pady=5)
        self.delta = ctk.CTkEntry(budget_frame, placeholder_text="1e-5")
        self.delta.pack(fill=tk.X, padx=10, pady=5)
        
        # Mechanism selection
        mechanism_frame = ctk.CTkFrame(dp_frame)
        mechanism_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(mechanism_frame, text="DP Mechanism:").pack(anchor="w", padx=10, pady=5)
        self.dp_mechanism = ctk.CTkComboBox(
            mechanism_frame,
            values=["Laplace", "Gaussian", "Exponential", "Random Response"]
        )
        self.dp_mechanism.pack(fill=tk.X, padx=10, pady=5)
        self.dp_mechanism.set("Laplace")
        
    def setup_sdg_config(self):
        """Setup Synthetic Data Generation configuration"""
        tab = self.technique_notebook.tab("Synthetic Data Generation")
        
        sdg_frame = ctk.CTkFrame(tab)
        sdg_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Model selection
        model_frame = ctk.CTkFrame(sdg_frame)
        model_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(model_frame, text="Generative Model:").pack(anchor="w", padx=10, pady=5)
        self.sdg_model = ctk.CTkComboBox(
            model_frame,
            values=["GAN", "VAE", "Bayesian Network", "Copula"]
        )
        self.sdg_model.pack(fill=tk.X, padx=10, pady=5)
        self.sdg_model.set("GAN")
        
        # Training parameters
        training_frame = ctk.CTkFrame(sdg_frame)
        training_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ctk.CTkLabel(training_frame, text="Training Epochs:").pack(anchor="w", padx=10, pady=2)
        self.training_epochs = ctk.CTkEntry(training_frame, placeholder_text="100")
        self.training_epochs.pack(fill=tk.X, padx=10, pady=2)
        
        ctk.CTkLabel(training_frame, text="Batch Size:").pack(anchor="w", padx=10, pady=2)
        self.batch_size = ctk.CTkEntry(training_frame, placeholder_text="32")
        self.batch_size.pack(fill=tk.X, padx=10, pady=2)
        
        ctk.CTkLabel(training_frame, text="Learning Rate:").pack(anchor="w", padx=10, pady=2)
        self.learning_rate = ctk.CTkEntry(training_frame, placeholder_text="0.001")
        self.learning_rate.pack(fill=tk.X, padx=10, pady=2)
        
    def setup_utility_analysis_tab(self):
        """Setup utility analysis configuration tab"""
        tab = self.notebook.tab("Utility Analysis")
        
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Utility metrics selection
        metrics_frame = ctk.CTkFrame(scroll_frame)
        metrics_frame.pack(fill=tk.X, padx=5, pady=10)
        
        metrics_label = ctk.CTkLabel(
            metrics_frame,
            text="Utility Metrics",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        metrics_label.pack(pady=10)
        
        # Statistical utility metrics
        stat_frame = ctk.CTkFrame(metrics_frame)
        stat_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(stat_frame, text="Statistical Utility:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10)
        
        self.utility_metrics = {}
        stat_metrics = ["Mean Absolute Error", "Root Mean Square Error", "Correlation Preservation", "KL Divergence"]
        
        for metric in stat_metrics:
            var = ctk.CTkCheckBox(stat_frame, text=metric)
            var.pack(anchor="w", padx=20, pady=2)
            var.select()
            self.utility_metrics[metric] = var
            
        # Analytical utility metrics
        anal_frame = ctk.CTkFrame(metrics_frame)
        anal_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(anal_frame, text="Analytical Utility:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10)
        
        anal_metrics = ["Query Accuracy", "Model Performance", "Hypothesis Testing"]
        
        for metric in anal_metrics:
            var = ctk.CTkCheckBox(anal_frame, text=metric)
            var.pack(anchor="w", padx=20, pady=2)
            var.select()
            self.utility_metrics[metric] = var
            
        # Run button
        run_button = ctk.CTkButton(
            scroll_frame,
            text="Run Utility Analysis",
            command=self.run_utility_analysis,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        run_button.pack(pady=20)
        
    def setup_reports_tab(self):
        """Setup reports configuration tab"""
        tab = self.notebook.tab("Reports")
        
        scroll_frame = ctk.CTkScrollableFrame(tab)
        scroll_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Report types
        report_frame = ctk.CTkFrame(scroll_frame)
        report_frame.pack(fill=tk.X, padx=5, pady=10)
        
        report_label = ctk.CTkLabel(
            report_frame,
            text="Report Generation",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        report_label.pack(pady=10)
        
        # Report type selection
        type_frame = ctk.CTkFrame(report_frame)
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(type_frame, text="Report Type:").pack(anchor="w", padx=10, pady=5)
        self.report_type = ctk.CTkComboBox(
            type_frame,
            values=["Executive Summary", "Technical Report", "Compliance Report", "Full Assessment"]
        )
        self.report_type.pack(fill=tk.X, padx=10, pady=5)
        self.report_type.set("Full Assessment")
        
        # Report format
        format_frame = ctk.CTkFrame(report_frame)
        format_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ctk.CTkLabel(format_frame, text="Output Format:").pack(anchor="w", padx=10, pady=5)
        self.report_format = ctk.CTkComboBox(
            format_frame,
            values=["PDF", "HTML", "JSON"]
        )
        self.report_format.pack(fill=tk.X, padx=10, pady=5)
        self.report_format.set("PDF")
        
        # Generate button
        generate_button = ctk.CTkButton(
            scroll_frame,
            text="Generate Report",
            command=self.generate_report,
            height=40,
            font=ctk.CTkFont(size=14, weight="bold")
        )
        generate_button.pack(pady=20)
        
    def toggle_qi_selection(self):
        """Toggle quasi-identifier selection mode"""
        if self.auto_detect_qi.get():
            self.qi_listbox_frame.pack_forget()
        else:
            self.qi_listbox_frame.pack(fill=tk.X, padx=10, pady=5)
            
    def toggle_suppression_config(self):
        """Toggle suppression configuration visibility"""
        if self.use_suppression.get():
            self.suppression_config.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.suppression_config.pack_forget()
            
    def toggle_generalization_config(self):
        """Toggle generalization configuration visibility"""
        if self.use_generalization.get():
            self.generalization_config.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.generalization_config.pack_forget()
            
    def toggle_perturbation_config(self):
        """Toggle perturbation configuration visibility"""
        if self.use_perturbation.get():
            self.perturbation_config.pack(fill=tk.X, padx=10, pady=5)
        else:
            self.perturbation_config.pack_forget()
            
    def update_individual_threshold(self, value):
        """Update individual risk threshold display"""
        self.individual_threshold_label.configure(text=f"{float(value):.2f}")
        
    def update_global_threshold(self, value):
        """Update global risk threshold display"""
        self.global_threshold_label.configure(text=f"{float(value):.2f}")
        
    def get_risk_config(self):
        """Get risk assessment configuration"""
        config = {
            'auto_detect_qi': self.auto_detect_qi.get(),
            'attack_types': [name for name, var in self.attack_vars.items() if var.get()],
            'individual_threshold': self.individual_threshold.get(),
            'global_threshold': self.global_threshold.get(),
            'quasi_identifiers': []  # Will be populated based on dataset
        }
        return config
        
    def get_privacy_config(self):
        """Get privacy enhancement configuration"""
        config = {
            'sdc': {
                'use_suppression': self.use_suppression.get(),
                'suppression_threshold': int(self.suppression_threshold.get() or 5),
                'use_generalization': self.use_generalization.get(),
                'generalization_levels': int(self.generalization_levels.get() or 2),
                'use_perturbation': self.use_perturbation.get(),
                'noise_level': self.noise_level.get()
            },
            'dp': {
                'epsilon': float(self.epsilon.get() or 1.0),
                'delta': float(self.delta.get() or 1e-5),
                'mechanism': self.dp_mechanism.get()
            },
            'sdg': {
                'model': self.sdg_model.get(),
                'epochs': int(self.training_epochs.get() or 100),
                'batch_size': int(self.batch_size.get() or 32),
                'learning_rate': float(self.learning_rate.get() or 0.001)
            }
        }
        return config
        
    def get_utility_config(self):
        """Get utility analysis configuration"""
        config = {
            'metrics': [name for name, var in self.utility_metrics.items() if var.get()]
        }
        return config
        
    def run_risk_assessment(self):
        """Run risk assessment with current configuration"""
        config = self.get_risk_config()
        self.main_window.run_risk_assessment(config)
        
    def run_privacy_enhancement(self):
        """Run privacy enhancement with current configuration"""
        config = self.get_privacy_config()
        self.main_window.run_privacy_enhancement(config)
        
    def run_utility_analysis(self):
        """Run utility analysis with current configuration"""
        config = self.get_utility_config()
        self.main_window.run_utility_analysis(config)
        
    def generate_report(self):
        """Generate report with current configuration"""
        report_type = self.report_type.get()
        self.main_window.generate_report(report_type)
        
    def show_risk_assessment(self):
        """Show risk assessment tab"""
        self.show()
        self.notebook.set("Risk Assessment")
        
    def show_privacy_enhancement(self):
        """Show privacy enhancement tab"""
        self.show()
        self.notebook.set("Privacy Enhancement")
        
    def show_utility_analysis(self):
        """Show utility analysis tab"""
        self.show()
        self.notebook.set("Utility Analysis")
        
    def show_reports(self):
        """Show reports tab"""
        self.show()
        self.notebook.set("Reports")
        
    def show(self):
        """Show the configuration panel"""
        if not self.is_visible:
            self.frame.grid(row=0, column=0, sticky="nsew")
            self.is_visible = True
            
    def hide(self):
        """Hide the configuration panel"""
        if self.is_visible:
            self.frame.grid_forget()
            self.is_visible = False
