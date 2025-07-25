"""
Custom GUI widgets for SafeData Pipeline
Reusable components for professional interface
"""

import tkinter as tk
import customtkinter as ctk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import seaborn as sns
from threading import Thread
import time

from assets.styles import APP_STYLES

class MetricsCard(ctk.CTkFrame):
    """Custom metrics display card"""
    
    def __init__(self, parent, title, value, color=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.color = color or APP_STYLES["primary_color"]
        
        self.setup_card()
        self.update_value(value)
        
    def setup_card(self):
        """Setup card layout"""
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Title
        self.title_label = ctk.CTkLabel(
            self,
            text=self.title,
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color="gray"
        )
        self.title_label.grid(row=0, column=0, pady=(10, 5), sticky="ew")
        
        # Value
        self.value_label = ctk.CTkLabel(
            self,
            text="--",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=self.color
        )
        self.value_label.grid(row=1, column=0, pady=(0, 10), sticky="ew")
        
    def update_value(self, value):
        """Update card value"""
        self.value_label.configure(text=str(value))

class ProgressDialog:
    """Progress dialog for long-running operations"""
    
    def __init__(self, parent, title, message):
        self.parent = parent
        self.title = title
        self.message = message
        self.is_open = False
        
    def show(self):
        """Show progress dialog"""
        if self.is_open:
            return
            
        self.dialog = ctk.CTkToplevel(self.parent)
        self.dialog.title(self.title)
        self.dialog.geometry("400x150")
        self.dialog.resizable(False, False)
        
        # Center the dialog
        self.dialog.transient(self.parent)
        self.dialog.grab_set()
        
        # Message
        message_label = ctk.CTkLabel(
            self.dialog,
            text=self.message,
            font=ctk.CTkFont(size=14)
        )
        message_label.pack(pady=20)
        
        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(self.dialog)
        self.progress_bar.pack(padx=40, pady=10, fill=tk.X)
        self.progress_bar.set(0)
        
        # Start animated progress
        self.animate_progress()
        
        self.is_open = True
        
    def animate_progress(self):
        """Animate progress bar"""
        def update_progress():
            progress = 0
            while self.is_open:
                progress = (progress + 0.1) % 1.0
                if self.is_open:
                    self.progress_bar.set(progress)
                time.sleep(0.1)
                
        self.progress_thread = Thread(target=update_progress, daemon=True)
        self.progress_thread.start()
        
    def close(self):
        """Close progress dialog"""
        if self.is_open:
            self.is_open = False
            self.dialog.destroy()

class StatusBar(ctk.CTkFrame):
    """Application status bar"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, height=30, **kwargs)
        
        self.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready",
            font=ctk.CTkFont(size=11),
            anchor="w"
        )
        self.status_label.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        
        # Time label
        self.time_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=11)
        )
        self.time_label.grid(row=0, column=1, padx=10, pady=5, sticky="e")
        
        self.update_time()
        
    def set_status(self, status):
        """Set status message"""
        self.status_label.configure(text=status)
        
    def update_time(self):
        """Update time display"""
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        self.time_label.configure(text=current_time)
        self.after(1000, self.update_time)

class PrivacyHeatmap(ctk.CTkFrame):
    """Privacy risk heatmap visualization"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.setup_heatmap()
        
    def setup_heatmap(self):
        """Setup heatmap canvas"""
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(6, 4))
        self.fig.patch.set_facecolor('#2b2b2b')
        
        # Initialize with empty heatmap
        data = np.random.rand(5, 5) * 0.1  # Low initial risk
        self.heatmap = sns.heatmap(
            data,
            annot=True,
            cmap="YlOrRd",
            ax=self.ax,
            cbar_kws={"label": "Privacy Risk"}
        )
        
        self.ax.set_title("Privacy Risk Heatmap", color='white')
        self.ax.set_xlabel("Attribute Groups", color='white')
        self.ax.set_ylabel("Record Groups", color='white')
        
        # Embed in canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def update_data(self, privacy_data):
        """Update heatmap with new privacy data"""
        if privacy_data:
            # Clear previous plot
            self.ax.clear()
            
            # Create new heatmap with privacy data
            # This is a simplified example - real implementation would use actual privacy metrics
            rows = privacy_data.get('rows', 5)
            cols = privacy_data.get('cols', 5)
            risk_matrix = np.random.rand(rows, cols) * 0.8  # Example risk data
            
            self.heatmap = sns.heatmap(
                risk_matrix,
                annot=True,
                fmt='.2f',
                cmap="YlOrRd",
                ax=self.ax,
                cbar_kws={"label": "Privacy Risk"}
            )
            
            self.ax.set_title("Privacy Risk Heatmap", color='white')
            self.ax.set_xlabel("Attribute Groups", color='white')
            self.ax.set_ylabel("Record Groups", color='white')
            
            # Refresh canvas
            self.canvas.draw()

class DataTable(ctk.CTkFrame):
    """Data table widget for displaying dataset information"""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.setup_table()
        
    def setup_table(self):
        """Setup table with scrollbars"""
        # Create treeview for table display
        self.tree_frame = ctk.CTkFrame(self)
        self.tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # This would be implemented with a proper table widget
        # For now, using a text widget as placeholder
        self.table_text = ctk.CTkTextbox(self.tree_frame)
        self.table_text.pack(fill=tk.BOTH, expand=True)
        
    def update_data(self, dataframe):
        """Update table with new data"""
        self.table_text.delete("0.0", tk.END)
        if dataframe is not None:
            # Display basic info about the dataset
            info_text = f"""Dataset Information:
Rows: {len(dataframe)}
Columns: {len(dataframe.columns)}
Memory Usage: {dataframe.memory_usage(deep=True).sum() / 1024**2:.2f} MB

Column Information:
{dataframe.dtypes.to_string()}

Sample Data:
{dataframe.head().to_string()}
"""
            self.table_text.insert("0.0", info_text)

class ParameterSlider(ctk.CTkFrame):
    """Custom parameter slider with label"""
    
    def __init__(self, parent, label, min_val, max_val, initial_val, callback=None, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.label = label
        self.callback = callback
        
        self.setup_slider(min_val, max_val, initial_val)
        
    def setup_slider(self, min_val, max_val, initial_val):
        """Setup slider with label"""
        # Label
        self.label_widget = ctk.CTkLabel(self, text=self.label)
        self.label_widget.pack(anchor="w", padx=5, pady=(5, 0))
        
        # Slider
        self.slider = ctk.CTkSlider(
            self,
            from_=min_val,
            to=max_val,
            number_of_steps=100,
            command=self.on_slider_change
        )
        self.slider.pack(fill=tk.X, padx=5, pady=2)
        self.slider.set(initial_val)
        
        # Value label
        self.value_label = ctk.CTkLabel(self, text=f"{initial_val:.3f}")
        self.value_label.pack(anchor="e", padx=5, pady=(0, 5))
        
    def on_slider_change(self, value):
        """Handle slider value change"""
        self.value_label.configure(text=f"{float(value):.3f}")
        if self.callback:
            self.callback(value)
            
    def get_value(self):
        """Get current slider value"""
        return self.slider.get()
        
    def set_value(self, value):
        """Set slider value"""
        self.slider.set(value)
        self.value_label.configure(text=f"{float(value):.3f}")

class AlertBanner(ctk.CTkFrame):
    """Alert banner for notifications"""
    
    def __init__(self, parent, message, alert_type="info", **kwargs):
        super().__init__(parent, **kwargs)
        
        self.alert_type = alert_type
        self.setup_banner(message)
        
    def setup_banner(self, message):
        """Setup alert banner"""
        # Color based on alert type
        colors = {
            "info": APP_STYLES["info_color"],
            "warning": APP_STYLES["warning_color"],
            "error": APP_STYLES["error_color"],
            "success": APP_STYLES["success_color"]
        }
        
        color = colors.get(self.alert_type, APP_STYLES["info_color"])
        
        # Icon and message
        self.grid_columnconfigure(1, weight=1)
        
        icon_text = {
            "info": "ℹ",
            "warning": "⚠",
            "error": "✗",
            "success": "✓"
        }
        
        icon_label = ctk.CTkLabel(
            self,
            text=icon_text.get(self.alert_type, "ℹ"),
            font=ctk.CTkFont(size=16),
            text_color=color
        )
        icon_label.grid(row=0, column=0, padx=10, pady=10)
        
        message_label = ctk.CTkLabel(
            self,
            text=message,
            font=ctk.CTkFont(size=12),
            text_color=color
        )
        message_label.grid(row=0, column=1, padx=5, pady=10, sticky="w")
        
        # Close button
        close_button = ctk.CTkButton(
            self,
            text="×",
            width=30,
            height=30,
            command=self.close,
            fg_color="transparent",
            text_color=color
        )
        close_button.grid(row=0, column=2, padx=10, pady=10)
        
    def close(self):
        """Close the alert banner"""
        self.destroy()
