"""
UI Styles and Themes for SafeData Pipeline
Centralized styling definitions for consistent GUI appearance
"""

from typing import Dict, Any, Tuple
import tkinter as tk

# Color palette definitions
COLORS = {
    # Primary colors
    'primary_blue': '#3498db',
    'primary_dark': '#2c3e50',
    'primary_light': '#ecf0f1',
    
    # Accent colors
    'accent_blue': '#2980b9',
    'accent_green': '#27ae60',
    'accent_orange': '#f39c12',
    'accent_red': '#e74c3c',
    'accent_purple': '#9b59b6',
    
    # Neutral colors
    'white': '#ffffff',
    'light_gray': '#f8f9fa',
    'medium_gray': '#bdc3c7',
    'dark_gray': '#34495e',
    'black': '#2c3e50',
    
    # Status colors
    'success': '#27ae60',
    'warning': '#f39c12',
    'error': '#e74c3c',
    'info': '#3498db',
    
    # Background colors
    'bg_primary': '#ffffff',
    'bg_secondary': '#f8f9fa',
    'bg_dark': '#2c3e50',
    'bg_card': '#ffffff',
    'bg_hover': '#ecf0f1',
    
    # Text colors
    'text_primary': '#2c3e50',
    'text_secondary': '#7f8c8d',
    'text_light': '#bdc3c7',
    'text_white': '#ffffff',
    'text_muted': '#95a5a6',
    
    # Border colors
    'border_light': '#e9ecef',
    'border_medium': '#dee2e6',
    'border_dark': '#adb5bd'
}

# Dark theme color overrides
DARK_COLORS = {
    'bg_primary': '#1e1e1e',
    'bg_secondary': '#2d2d30',
    'bg_dark': '#252526',
    'bg_card': '#2d2d30',
    'bg_hover': '#3e3e42',
    
    'text_primary': '#cccccc',
    'text_secondary': '#969696',
    'text_light': '#6a6a6a',
    'text_white': '#ffffff',
    'text_muted': '#808080',
    
    'border_light': '#3e3e42',
    'border_medium': '#464647',
    'border_dark': '#5a5a5a'
}

# Font configurations
FONTS = {
    'default_family': 'Segoe UI',
    'fallback_families': ['Arial', 'Helvetica', 'sans-serif'],
    'monospace_family': 'Consolas',
    'monospace_fallback': ['Courier New', 'Monaco', 'monospace'],
    
    'sizes': {
        'tiny': 8,
        'small': 10,
        'normal': 11,
        'medium': 12,
        'large': 14,
        'xlarge': 16,
        'xxlarge': 18,
        'title': 20,
        'header': 24,
        'display': 32
    },
    
    'weights': {
        'light': 'light',
        'normal': 'normal',
        'bold': 'bold'
    }
}

# Spacing and sizing constants
SPACING = {
    'xs': 4,
    'sm': 8,
    'md': 12,
    'lg': 16,
    'xl': 20,
    'xxl': 24,
    'xxxl': 32
}

SIZES = {
    'button_height': 35,
    'input_height': 32,
    'header_height': 60,
    'sidebar_width': 250,
    'card_min_width': 200,
    'card_min_height': 150,
    'icon_small': 16,
    'icon_medium': 24,
    'icon_large': 32
}

# Application-specific style definitions
APP_STYLES = {
    # Core colors
    'primary_color': COLORS['primary_blue'],
    'secondary_color': COLORS['primary_dark'],
    'accent_color': COLORS['accent_blue'],
    'background_color': COLORS['bg_primary'],
    'surface_color': COLORS['bg_card'],
    'text_color': COLORS['text_primary'],
    
    # Status colors
    'success_color': COLORS['success'],
    'warning_color': COLORS['warning'],
    'error_color': COLORS['error'],
    'info_color': COLORS['info'],
    
    # Interactive elements
    'button_color': COLORS['primary_blue'],
    'button_hover': COLORS['accent_blue'],
    'button_disabled': COLORS['medium_gray'],
    'link_color': COLORS['primary_blue'],
    'link_hover': COLORS['accent_blue'],
    
    # Form elements
    'input_bg': COLORS['white'],
    'input_border': COLORS['border_medium'],
    'input_focus': COLORS['primary_blue'],
    'input_error': COLORS['error'],
    
    # Navigation
    'nav_bg': COLORS['primary_dark'],
    'nav_text': COLORS['text_white'],
    'nav_hover': COLORS['accent_blue'],
    'nav_active': COLORS['primary_blue'],
    
    # Cards and panels
    'card_bg': COLORS['bg_card'],
    'card_border': COLORS['border_light'],
    'card_shadow': 'rgba(0, 0, 0, 0.1)',
    'panel_bg': COLORS['bg_secondary'],
    
    # Data visualization
    'chart_primary': COLORS['primary_blue'],
    'chart_secondary': COLORS['accent_green'],
    'chart_tertiary': COLORS['accent_orange'],
    'chart_quaternary': COLORS['accent_purple'],
    'chart_grid': COLORS['border_light'],
    'chart_text': COLORS['text_secondary'],
    
    # Privacy-specific colors
    'privacy_high': COLORS['success'],
    'privacy_medium': COLORS['warning'],
    'privacy_low': COLORS['error'],
    'risk_low': COLORS['success'],
    'risk_medium': COLORS['warning'],
    'risk_high': COLORS['error'],
    'risk_critical': '#c0392b',
    
    # Utility colors
    'utility_excellent': COLORS['success'],
    'utility_good': '#2ecc71',
    'utility_fair': COLORS['warning'],
    'utility_poor': COLORS['error']
}

# Dark theme style overrides
DARK_THEME_STYLES = {
    'background_color': DARK_COLORS['bg_primary'],
    'surface_color': DARK_COLORS['bg_card'],
    'text_color': DARK_COLORS['text_primary'],
    'input_bg': DARK_COLORS['bg_card'],
    'input_border': DARK_COLORS['border_medium'],
    'card_bg': DARK_COLORS['bg_card'],
    'card_border': DARK_COLORS['border_light'],
    'panel_bg': DARK_COLORS['bg_secondary'],
    'chart_grid': DARK_COLORS['border_light'],
    'chart_text': DARK_COLORS['text_secondary']
}

# CustomTkinter specific styling
CTK_STYLES = {
    'corner_radius': 6,
    'border_width': 1,
    'button_corner_radius': 6,
    'button_border_width': 0,
    'entry_corner_radius': 6,
    'frame_corner_radius': 6,
    'progressbar_corner_radius': 6,
    'slider_corner_radius': 6,
    'switch_corner_radius': 12,
    'checkbox_corner_radius': 4,
    'radiobutton_corner_radius': 100,
    'scrollbar_corner_radius': 6
}

class StyleManager:
    """Manages application styling and theme switching"""
    
    def __init__(self, theme: str = 'light'):
        self.current_theme = theme
        self.styles = self._get_current_styles()
        self.callbacks = []
    
    def _get_current_styles(self) -> Dict[str, Any]:
        """Get styles for current theme"""
        if self.current_theme == 'dark':
            styles = APP_STYLES.copy()
            styles.update(DARK_THEME_STYLES)
            return styles
        else:
            return APP_STYLES.copy()
    
    def set_theme(self, theme: str):
        """Change application theme"""
        if theme in ['light', 'dark']:
            self.current_theme = theme
            self.styles = self._get_current_styles()
            self._notify_theme_change()
    
    def get_color(self, color_name: str) -> str:
        """Get color by name"""
        return self.styles.get(color_name, COLORS.get(color_name, '#000000'))
    
    def get_font(self, size: str = 'normal', weight: str = 'normal', family: str = None) -> Tuple[str, int, str]:
        """Get font configuration"""
        font_family = family or FONTS['default_family']
        font_size = FONTS['sizes'].get(size, FONTS['sizes']['normal'])
        font_weight = FONTS['weights'].get(weight, FONTS['weights']['normal'])
        
        return (font_family, font_size, font_weight)
    
    def get_spacing(self, size: str) -> int:
        """Get spacing value"""
        return SPACING.get(size, SPACING['md'])
    
    def get_size(self, element: str) -> int:
        """Get size value for UI element"""
        return SIZES.get(element, 32)
    
    def add_theme_change_callback(self, callback):
        """Add callback for theme changes"""
        self.callbacks.append(callback)
    
    def _notify_theme_change(self):
        """Notify all callbacks of theme change"""
        for callback in self.callbacks:
            try:
                callback(self.current_theme)
            except Exception:
                pass  # Ignore callback errors
    
    def get_chart_colors(self, count: int = 1) -> list:
        """Get color palette for charts"""
        base_colors = [
            self.get_color('chart_primary'),
            self.get_color('chart_secondary'),
            self.get_color('chart_tertiary'),
            self.get_color('chart_quaternary'),
            COLORS['accent_red'],
            COLORS['accent_purple'],
            '#1abc9c',
            '#e67e22',
            '#34495e',
            '#16a085'
        ]
        
        if count <= len(base_colors):
            return base_colors[:count]
        else:
            # Generate additional colors
            import colorsys
            additional_colors = []
            for i in range(count - len(base_colors)):
                hue = (i * 0.618033988749895) % 1  # Golden ratio for good distribution
                saturation = 0.7
                value = 0.8
                rgb = colorsys.hsv_to_rgb(hue, saturation, value)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
                )
                additional_colors.append(hex_color)
            
            return base_colors + additional_colors
    
    def get_risk_color(self, risk_value: float) -> str:
        """Get color based on risk level"""
        if risk_value < 0.1:
            return self.get_color('risk_low')
        elif risk_value < 0.3:
            return self.get_color('privacy_medium')
        elif risk_value < 0.7:
            return self.get_color('risk_high')
        else:
            return self.get_color('risk_critical')
    
    def get_utility_color(self, utility_value: float) -> str:
        """Get color based on utility level"""
        if utility_value > 0.8:
            return self.get_color('utility_excellent')
        elif utility_value > 0.6:
            return self.get_color('utility_good')
        elif utility_value > 0.4:
            return self.get_color('utility_fair')
        else:
            return self.get_color('utility_poor')
    
    def create_gradient(self, start_color: str, end_color: str, steps: int = 10) -> list:
        """Create color gradient between two colors"""
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        
        gradient = []
        for i in range(steps):
            ratio = i / (steps - 1)
            r = start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio
            g = start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio
            b = start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio
            gradient.append(rgb_to_hex((r, g, b)))
        
        return gradient

# Global style manager instance
style_manager = StyleManager()

# Convenience functions
def get_app_styles(theme: str = None) -> Dict[str, Any]:
    """Get application styles for specified theme"""
    if theme:
        temp_manager = StyleManager(theme)
        return temp_manager.styles
    return style_manager.styles

def get_color(color_name: str) -> str:
    """Get color value by name"""
    return style_manager.get_color(color_name)

def get_font(size: str = 'normal', weight: str = 'normal', family: str = None) -> Tuple[str, int, str]:
    """Get font configuration"""
    return style_manager.get_font(size, weight, family)

def get_spacing(size: str) -> int:
    """Get spacing value"""
    return style_manager.get_spacing(size)

def get_chart_colors(count: int = 1) -> list:
    """Get chart color palette"""
    return style_manager.get_chart_colors(count)

def apply_widget_style(widget, style_type: str = 'default'):
    """Apply styling to tkinter/customtkinter widget"""
    styles = style_manager.styles
    
    if style_type == 'button':
        if hasattr(widget, 'configure'):
            widget.configure(
                fg_color=styles['button_color'],
                hover_color=styles['button_hover'],
                corner_radius=CTK_STYLES['button_corner_radius'],
                border_width=CTK_STYLES['button_border_width']
            )
    elif style_type == 'entry':
        if hasattr(widget, 'configure'):
            widget.configure(
                fg_color=styles['input_bg'],
                border_color=styles['input_border'],
                corner_radius=CTK_STYLES['entry_corner_radius']
            )
    elif style_type == 'frame':
        if hasattr(widget, 'configure'):
            widget.configure(
                fg_color=styles['surface_color'],
                corner_radius=CTK_STYLES['frame_corner_radius']
            )
    elif style_type == 'label':
        if hasattr(widget, 'configure'):
            widget.configure(
                text_color=styles['text_color']
            )

# CSS styles for HTML reports
HTML_STYLES = """
<style>
    :root {
        --primary-color: %s;
        --secondary-color: %s;
        --accent-color: %s;
        --background-color: %s;
        --text-color: %s;
        --success-color: %s;
        --warning-color: %s;
        --error-color: %s;
        --border-color: %s;
    }
    
    body {
        font-family: %s;
        color: var(--text-color);
        background-color: var(--background-color);
        line-height: 1.6;
        margin: 0;
        padding: 20px;
    }
    
    .container {
        max-width: 1200px;
        margin: 0 auto;
        background: white;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .header {
        background: var(--primary-color);
        color: white;
        padding: 30px;
        text-align: center;
    }
    
    .metric-card {
        background: #f8f9fa;
        border-left: 4px solid var(--primary-color);
        padding: 20px;
        margin: 15px;
        border-radius: 6px;
    }
    
    .success { border-left-color: var(--success-color); }
    .warning { border-left-color: var(--warning-color); }
    .error { border-left-color: var(--error-color); }
    
    .btn {
        background: var(--primary-color);
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 6px;
        cursor: pointer;
        font-size: 14px;
    }
    
    .btn:hover {
        background: var(--accent-color);
    }
    
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
    }
    
    th, td {
        padding: 12px;
        text-align: left;
        border-bottom: 1px solid var(--border-color);
    }
    
    th {
        background: var(--secondary-color);
        color: white;
    }
</style>
""" % (
    APP_STYLES['primary_color'],
    APP_STYLES['secondary_color'],
    APP_STYLES['accent_color'],
    APP_STYLES['background_color'],
    APP_STYLES['text_color'],
    APP_STYLES['success_color'],
    APP_STYLES['warning_color'],
    APP_STYLES['error_color'],
    COLORS['border_medium'],
    FONTS['default_family']
)

# Export commonly used style configurations
__all__ = [
    'APP_STYLES', 'COLORS', 'FONTS', 'SPACING', 'SIZES',
    'StyleManager', 'style_manager',
    'get_app_styles', 'get_color', 'get_font', 'get_spacing', 'get_chart_colors',
    'apply_widget_style', 'HTML_STYLES'
]
