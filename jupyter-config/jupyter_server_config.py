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

# Disable token and password authentication so JupyterLab opens directly
# without any login screen. This is safe because ModelMind only listens on
# 127.0.0.1 (localhost) and is never exposed to the network.
c.IdentityProvider.token = ""
c.IdentityProvider.password = ""
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
