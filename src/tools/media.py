import shutil
import subprocess
import re

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


def control_media_app(app_name: str, action: str) -> str:
    """Control only the explicitly named MPRIS player; never use playerctl's default player."""
    if not shutil.which("playerctl"):
        return "playerctl utility is not installed."

    app_name = (app_name or "").strip()
    action = (action or "").strip().lower()
    allowed_actions = {"play", "pause", "play-pause", "next", "previous", "stop"}
    if not app_name:
        return "Specify the application whose playback should be controlled."
    if action not in allowed_actions:
        return "Unsupported media action. Choose play, pause, play-pause, next, previous, or stop."

    try:
        listed = subprocess.run(
            ["playerctl", "--list-all"], capture_output=True, text=True, timeout=3
        )
        players = [line.strip() for line in listed.stdout.splitlines() if line.strip()]
        normalize = lambda value: re.sub(r"[^a-z0-9]+", "", value.lower())
        requested = normalize(app_name)
        matches = [
            player for player in players
            if normalize(player) == requested
            or normalize(player.split(".", 1)[0]) == requested
        ]
        if not matches:
            return f"No active MPRIS player matched {app_name!r}; playback was not changed."
        if len(matches) > 1:
            return f"Multiple {app_name} players are active ({', '.join(matches)}); playback was not changed."

        player = matches[0]
        result = subprocess.run(
            ["playerctl", f"--player={player}", action],
            capture_output=True, text=True, timeout=3,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            return f"Could not {action} {app_name}: {detail or 'player rejected the command'}."
        return f"Sent {action} to {app_name} ({player}) only."
    except subprocess.TimeoutExpired:
        return f"Timed out while controlling {app_name}; playback was not changed."
    except Exception as e:
        return f"Unable to control {app_name}: {e}"
