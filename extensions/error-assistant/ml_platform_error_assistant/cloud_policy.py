import os

DEFAULT_CLOUD_FALLBACK_MODE = "disabled"
VALID_CLOUD_FALLBACK_MODES = {"disabled", "local_only", "enabled"}


def cloud_fallback_mode():
    """Return the configured cloud fallback mode.

    Gemini and other remote providers remain explicitly disabled unless a
    conscious deployment decision opts into a cloud route.
    """
    mode = os.getenv("ML_PLATFORM_CLOUD_FALLBACK", DEFAULT_CLOUD_FALLBACK_MODE)
    normalized = str(mode).strip().lower()
    if normalized not in VALID_CLOUD_FALLBACK_MODES:
        return DEFAULT_CLOUD_FALLBACK_MODE
    return normalized


def gemini_fallback_enabled():
    return cloud_fallback_mode() == "enabled"


def gemini_api_key_configured():
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def health_policy_status():
    mode = cloud_fallback_mode()
    return {
        "cloud_fallback": mode,
        "gemini_fallback_available": mode == "enabled" and gemini_api_key_configured(),
    }
