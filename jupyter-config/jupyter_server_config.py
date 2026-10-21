from pathlib import Path


logo_path = (
    Path(__file__).resolve().parents[1]
    / "extensions"
    / "error-assistant"
    / "style"
    / "modelmind-logo.png"
)

c.Branding.logo_uri = str(logo_path)
c.Branding.splash_logo_uri = str(logo_path)
c.Branding.short_name = "ModelMind Notebook"

c.ServerApp.token = ""
c.ServerApp.password = ""
c.ServerApp.open_browser = False
c.ExtensionApp.open_browser = False
c.LabServerApp.open_browser = False

import sys
ext_dir = str(Path(__file__).resolve().parents[1] / "extensions" / "error-assistant")
if ext_dir not in sys.path:
    sys.path.insert(0, ext_dir)

c.ServerApp.jpserver_extensions = {"ml_platform_error_assistant": True}
