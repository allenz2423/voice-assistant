import shutil
import subprocess

def get_now_playing() -> str:
    """Inspects currently playing media track title, artist, and status via playerctl."""
    if not shutil.which("playerctl"):
        return "playerctl utility is not installed."

    try:
        res = subprocess.run(
            ["playerctl", "metadata", "--format", "{{playerName}}: {{title}} by {{artist}} ({{status}})"],
            capture_output=True,
            text=True,
            timeout=2
        )
        out = res.stdout.strip()
        if out:
            return f"Currently playing: {out}"
        else:
            # Check playerctl status directly
            st = subprocess.run(["playerctl", "status"], capture_output=True, text=True, timeout=2).stdout.strip()
            if st:
                return f"Media player is currently {st.lower()}, but no track metadata is available."
            return "No active media playback detected."
    except Exception as e:
        return f"Unable to check media playback: {e}"

def media_control(action: str) -> str:
    """Controls media playback using playerctl (play, pause, play-pause, next, previous, stop)."""
    act = (action or "play-pause").strip().lower()
    if not shutil.which("playerctl"):
        return "playerctl utility is not installed."

    valid_actions = {
        "play": "play",
        "pause": "pause",
        "play-pause": "play-pause",
        "toggle": "play-pause",
        "next": "next",
        "previous": "previous",
        "stop": "stop"
    }
    cmd_act = valid_actions.get(act, "play-pause")
    subprocess.run(["playerctl", cmd_act], check=False)
    return f"Media control '{cmd_act}' executed."
