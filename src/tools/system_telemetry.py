import os
import shutil
import subprocess
from typing import Optional

def get_system_status() -> str:
    """Dynamically queries CPU, RAM, storage, and all installed GPUs without hardcoded models."""
    parts = []

    # 1. CPU Load & Cores
    try:
        with open("/proc/loadavg", "r") as f:
            load = f.read().split()[:3]
        with open("/proc/cpuinfo", "r") as f:
            cores = len([line for line in f if line.startswith("processor")])
        parts.append(f"CPU has {cores} logical cores with load average {load[0]}, {load[1]}, {load[2]}.")
    except Exception:
        pass

    # 2. RAM Usage
    try:
        mem = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                p = line.split(":")
                if len(p) == 2:
                    mem[p[0].strip()] = int(p[1].strip().split()[0])
        total_gb = mem.get("MemTotal", 0) / (1024 * 1024)
        avail_gb = mem.get("MemAvailable", 0) / (1024 * 1024)
        used_gb = total_gb - avail_gb
        used_pct = (used_gb / total_gb * 100) if total_gb else 0
        parts.append(f"Memory is {used_pct:.0f} percent in use ({used_gb:.1f} gigabytes used out of {total_gb:.1f} gigabytes).")
    except Exception:
        pass

    # 3. Disk Storage
    try:
        st = os.statvfs("/")
        total_disk = (st.f_blocks * st.f_frsize) / (1024 ** 3)
        avail_disk = (st.f_bavail * st.f_frsize) / (1024 ** 3)
        used_disk = total_disk - avail_disk
        parts.append(f"Root storage has {avail_disk:.1f} gigabytes free out of {total_disk:.1f} gigabytes.")
    except Exception:
        pass

    # 4. GPU Telemetry (NVIDIA, AMD, or generic sysfs)
    gpu_found = False
    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run([
                "nvidia-smi",
                "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits"
            ], capture_output=True, text=True, timeout=2)
            if res.returncode == 0 and res.stdout.strip():
                gpu_lines = []
                for line in res.stdout.strip().splitlines():
                    idx, name, temp, util, mem_used, mem_tot = [x.strip() for x in line.split(",")]
                    used_g = int(mem_used) / 1024
                    tot_g = int(mem_tot) / 1024
                    gpu_lines.append(f"GPU {idx} ({name}) is at {temp} degrees Celsius with {used_g:.1f} of {tot_g:.1f} gigabytes VRAM in use")
                parts.append(" ".join(gpu_lines) + ".")
                gpu_found = True
        except Exception:
            pass

    if not gpu_found and shutil.which("rocm-smi"):
        try:
            res = subprocess.run(["rocm-smi", "--showtemp", "--showuse"], capture_output=True, text=True, timeout=2)
            if res.returncode == 0:
                parts.append("AMD GPU active (queried via rocm-smi).")
                gpu_found = True
        except Exception:
            pass

    if not parts:
        return "Unable to collect system status."

    return " ".join(parts)

def list_processes(sort_by: Optional[str] = "cpu", limit: int = 5) -> str:
    """Lists top running processes sorted by CPU or memory consumption."""
    sort_flag = "-%mem" if (sort_by or "").lower() == "memory" else "-%cpu"
    num = max(1, min(limit, 20))
    try:
        res = subprocess.run(
            ["ps", "-eo", "pid,%cpu,%mem,comm", f"--sort={sort_flag}"],
            capture_output=True,
            text=True,
            timeout=2
        )
        lines = res.stdout.strip().splitlines()
        header = lines[0] if lines else ""
        top_lines = lines[1:num + 1]
        formatted = []
        for l in top_lines:
            p = l.split(None, 3)
            if len(p) >= 4:
                pid, cpu, mem, comm = p[0], p[1], p[2], p[3]
                formatted.append(f"{comm} (PID {pid}): {cpu}% CPU, {mem}% RAM")
        return f"Top processes by {sort_by or 'cpu'}: " + ", ".join(formatted)
    except Exception as e:
        return f"Error listing processes: {e}"

def kill_process(target: str, force: bool = False) -> str:
    """Terminates a process by PID or name."""
    t = (target or "").strip()
    if not t:
        return "Please specify a PID or process name to terminate."

    flag = "-9" if force else "-15"
    if t.isdigit():
        subprocess.run(["kill", flag, t], check=False)
        return f"Sent termination signal to PID {t}."
    else:
        subprocess.run(["pkill", flag, "-f", t], check=False)
        return f"Sent termination signal to processes matching '{t}'."

def list_audio_devices() -> str:
    """Dynamically discovers and lists all audio playback sinks and capture sources."""
    sinks = []
    sources = []

    if shutil.which("pactl"):
        try:
            s_raw = subprocess.run(["pactl", "list", "sinks", "short"], capture_output=True, text=True).stdout
            for l in s_raw.splitlines():
                p = l.split()
                if len(p) >= 2:
                    sinks.append(p[1])
            src_raw = subprocess.run(["pactl", "list", "sources", "short"], capture_output=True, text=True).stdout
            for l in src_raw.splitlines():
                p = l.split()
                if len(p) >= 2 and not p[1].endswith(".monitor"):
                    sources.append(p[1])
        except Exception:
            pass

    out = []
    if sinks:
        out.append(f"Available Audio Sinks ({len(sinks)}): {', '.join(sinks[:6])}")
    if sources:
        out.append(f"Available Audio Sources ({len(sources)}): {', '.join(sources[:6])}")

    return "\n".join(out) if out else "No audio devices detected via PulseAudio/PipeWire."

def volume_control(action: str, level: Optional[int] = None) -> str:
    """Controls volume and mute status for default sink or microphone."""
    act = (action or "up").strip().lower()

    if shutil.which("wpctl"):
        if act == "up":
            step = f"{level or 5}%+"
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", step], check=False)
            return "Volume increased."
        elif act == "down":
            step = f"{level or 5}%-"
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", step], check=False)
            return "Volume decreased."
        elif act == "set" and level is not None:
            clamped = max(0, min(level, 100)) / 100.0
            subprocess.run(["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{clamped:.2f}"], check=False)
            return f"Volume set to {level} percent."
        elif act in ["mute", "unmute", "toggle-mute"]:
            subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "toggle"], check=False)
            return "Audio mute toggled."
        elif act in ["mute-mic", "unmute-mic", "toggle-mic"]:
            subprocess.run(["wpctl", "set-mute", "@DEFAULT_AUDIO_SOURCE@", "toggle"], check=False)
            return "Microphone mute toggled."

    elif shutil.which("pactl"):
        if act == "up":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "+5%"], check=False)
            return "Volume increased."
        elif act == "down":
            subprocess.run(["pactl", "set-sink-volume", "@DEFAULT_SINK@", "-5%"], check=False)
            return "Volume decreased."
        elif act in ["mute", "unmute", "toggle-mute"]:
            subprocess.run(["pactl", "set-sink-mute", "@DEFAULT_SINK@", "toggle"], check=False)
            return "Audio mute toggled."

    return "Audio volume control is not available on this system."
