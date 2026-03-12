from rich.theme import Theme
from rich.console import Console
import os

def get_current_theme_name() -> str:
    try:
        from settings_manager import SettingsManager
        return SettingsManager.load().get("theme", "dark").lower()
    except Exception:
        return "dark"

class ThemeMeta(type):
    @property
    def PRIMARY(cls): return cls.get_palette()["primary"]
    @property
    def ACCENT(cls): return cls.get_palette()["accent"]
    @property
    def SECONDARY(cls): return cls.get_palette()["secondary"]
    @property
    def HIGHLIGHT(cls): return cls.get_palette()["highlight"]
    @property
    def BOX_STYLE(cls): return "rounded"

class ThemeEngine(metaclass=ThemeMeta):
    """
    Centralized theme management for SelfResearch OS.
    Features dynamic palette loading for a deep, premium UX.
    """

    # --- PALETTE DEFINITIONS ---
    PALETTES = {
        "dark": {
            "primary": "#00ff87",      # Spring Green
            "accent": "#00ffff",       # Cyan
            "secondary": "#007bff",    # Dodger Blue
            "highlight": "#bd93f9",    # Plum
            "border": "#00ffff",
            "warning": "#ffb86c",      # Orange
            "danger": "#ff5555",       # Red
            "info": "#8be9fd",         # Purple/Cyan
            "orchestrator": "#ff79c6", # Pink
            "proxy": "#50fa7b",
            "bg": "#121212",
            "text": "#f8f8f2",
            "dim": "#6272a4",
            "success": "#50fa7b"
        },
        "matrix": {
            "primary": "#00ff00",
            "accent": "#00cc00",
            "secondary": "#009900",
            "highlight": "#00ffcc",
            "border": "#00ff66",
            "warning": "#ccff00",
            "danger": "#ff0000",
            "info": "#00ff66",
            "orchestrator": "#33ff33",
            "proxy": "#66ff66",
            "bg": "#000000",
            "text": "#00ff00",
            "dim": "#004400",
            "success": "#00ff00"
        },
        "synthwave": {
            "primary": "#ff71ce",
            "accent": "#f39c12",
            "secondary": "#36d7b7",
            "highlight": "#a29bfe",
            "border": "#ff71ce",
            "warning": "#f1c40f",
            "danger": "#e74c3c",
            "info": "#00cec9",
            "orchestrator": "#fd79a8",
            "proxy": "#55efc4",
            "bg": "#2b213a",
            "text": "#f8f8f2",
            "dim": "#6c5ce7",
            "success": "#00b894"
        },
        "dracula": {
            "primary": "#ff79c6",
            "accent": "#bd93f9",
            "secondary": "#8be9fd",
            "highlight": "#50fa7b",
            "border": "#bd93f9",
            "warning": "#f1fa8c",
            "danger": "#ff5555",
            "info": "#8be9fd",
            "orchestrator": "#ffb86c",
            "proxy": "#50fa7b",
            "bg": "#282a36",
            "text": "#f8f8f2",
            "dim": "#6272a4",
            "success": "#50fa7b"
        },
        "cyberpunk": {
            "primary": "#f1fa8c",      # Yellow
            "accent": "#ff79c6",       # Pink
            "secondary": "#bd93f9",    # Purple
            "highlight": "#8be9fd",    # Cyan
            "border": "#ff79c6",
            "warning": "#ffb86c",
            "danger": "#ff5555",
            "info": "#f1fa8c",
            "orchestrator": "#ff79c6",
            "proxy": "#50fa7b",
            "bg": "#282a36",
            "text": "#f8f8f2",
            "dim": "#6272a4",
            "success": "#50fa7b"
        },
        "light": {
            "primary": "#005f00",
            "accent": "#005f5f",
            "secondary": "#0000af",
            "highlight": "#8700af",
            "border": "#005f5f",
            "warning": "#af5f00",
            "danger": "#af0000",
            "info": "#008787",
            "orchestrator": "#af00d7",
            "proxy": "#008700",
            "bg": "#ffffff",
            "text": "#1c1c1c",
            "dim": "#808080",
            "success": "#008700"
        },
        "high-contrast": {
            "primary": "#ffffff",
            "accent": "#ffff00",
            "secondary": "#00ffff",
            "highlight": "#ff00ff",
            "border": "#ffffff",
            "warning": "#ff8800",
            "danger": "#ff0000",
            "info": "#00ffff",
            "orchestrator": "#ff00ff",
            "proxy": "#ffff00",
            "bg": "#000000",
            "text": "#ffffff",
            "dim": "#aaaaaa",
            "success": "#00ff00"
        }
    }

    @classmethod
    def get_palette(cls) -> dict:
        name = get_current_theme_name()
        return cls.PALETTES.get(name, cls.PALETTES["dark"])

    @classmethod
    def get_theme(cls) -> Theme:
        p = cls.get_palette()
        return Theme({
            "os.header": f"bold {p['primary']}",
            "os.header_accent": f"bold {p['accent']}",
            "os.sidebar.label": f"bold {p['primary']}",
            "os.sidebar.value": p['warning'],
            "os.workspace.title": f"bold {p['primary']}",
            "os.workspace.subtitle": f"italic {p['secondary']}",
            "os.workspace.text": p['text'],
            "os.console.title": f"bold {p['highlight']}",
            "os.console.log": f"{p['dim']}",
            "os.border": f"bold {p['border']}",
            "os.status.success": f"bold {p['success']}",
            "os.status.error": f"bold {p['danger']}",
            "os.status.warning": f"bold {p['warning']}",
            "os.status.info": f"bold {p['info']}",
            "os.branding": f"bold {p['secondary']}",
            "os.project": f"bold {p['highlight']}",
            "os.mode": f"bold {p['warning']}",
            "os.progress.description": f"bold {p['info']}",
            "os.progress.bar": p['highlight'],
            "os.progress.spinner": p['accent'],
            "os.orchestrator.title": f"bold {p['orchestrator']}",
            "os.orchestrator.tidbit": f"italic {p['orchestrator']}",
            "os.proxy.title": f"bold {p['proxy']}",
            "os.proxy.content": f"bold {p['proxy']}",
            "os.sota.title": f"bold {p['accent']}",
            "os.input.prompt": f"bold {p['primary']}",
            "os.tool.name": f"bold {p['secondary']}",
            "os.tool.args": f"italic {p['info']}"
        })

    @classmethod
    def get_pt_style(cls) -> dict:
        """Returns prompt_toolkit style dict matching the active Rich theme."""
        p = cls.get_palette()
        return {
            'completion-menu.completion': f"bg:{p['bg']} fg:{p['primary']}",
            'completion-menu.completion.current': f"bg:{p['primary']} fg:{p['bg']} bold",
            'completion-menu.meta.completion': f"bg:{p['bg']} fg:{p['dim']} italic",
            'completion-menu.meta.completion.current': f"bg:{p['primary']} fg:{p['bg']} italic",
            'scrollbar.background': f"bg:{p['bg']}",
            'scrollbar.button': f"bg:{p['secondary']}",
            'frame.border': f"fg:{p['border']}",
            'prompt': f"fg:{p['primary']} bold",
            'prompt.symbol': f"fg:{p['warning']}",
            'prompt.name': f"bg:{p['warning']} fg:{p['bg']} bold",
        }

    @staticmethod
    def get_console() -> Console:
        is_windows = os.name == 'nt'
        is_windows_terminal = os.environ.get('WT_SESSION') is not None
        color_system = "256" if (is_windows and not is_windows_terminal) else "truecolor"
            
        return Console(
            theme=ThemeEngine.get_theme(),
            color_system=color_system,
            force_terminal=True,
            legacy_windows=is_windows and not is_windows_terminal,
            width=None
        )

theme_console = ThemeEngine.get_console()
