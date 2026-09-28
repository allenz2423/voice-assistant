#!/usr/bin/env python3
import sys
import time
import re
import numpy as np
import sounddevice as sd
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load Faster-Whisper
print("[1/3] Loading faster-whisper...")
from src.stt.transcriber import WhisperTranscriber
stt = WhisperTranscriber(model_size="base.en", device="cuda", device_index=0, compute_type="int8_float32")

# Wake word matcher
wake_regex = re.compile(r"\b(hey\b[,.\s]*)?(adam|shin)\b[,.:!?\s]*", re.IGNORECASE)

print("\n" + "=" * 60)
print(" BARE-BONES MICROPHONE & WHISPER TRANSCRIPTION TEST")
print("=" * 60)
print("Get ready to speak into your microphone...")
for i in [3, 2, 1]:
    print(f"Recording in {i}...")
    time.sleep(1)

print("\n>>> RECORDING NOW! Say: 'Hey Adam, what time is it?' <<<")
sample_rate = 16000
duration = 4.0  # 4 seconds

try:
    audio = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype="float32", device="pulse")
    sd.wait()
except Exception as e:
    print(f"Error recording from pulse device: {e}")
    sys.exit(1)

audio_flat = audio.flatten()
rms = float(np.sqrt(np.mean(audio_flat**2)))
peak = float(np.max(np.abs(audio_flat)))

print(f"\n[Audio Stats] RMS Level: {rms:.6f} | Peak Amplitude: {peak:.6f}")
if peak < 0.001:
    print("[WARNING] Very low audio level detected! Is your microphone input gain turned up?")
else:
    print("[OK] Microphone signal detected!")

print("\n[2/3] Transcribing with Faster-Whisper...")
t0 = time.time()
text = stt.transcribe(audio_flat)
elapsed = time.time() - t0

print("-" * 60)
print(f"Transcription ({elapsed:.2f}s): \"{text}\"")
print("-" * 60)

if text:
    match = wake_regex.search(text.strip())
    if match:
        cmd = text.strip()[match.end():].strip()
        print(f"[Wake Word] SUCCESS! Matched wake word! Trailing command: \"{cmd}\"")
    else:
        print(f"[Wake Word] Speech heard, but did not match 'Hey Adam' or 'Hey Shin'.")
else:
    print("[Whisper] No speech detected in the recording.")

print("\n[3/3] Testing TTS Playback...")
try:
    import subprocess
    msg = f"I heard you say: {text}" if text else "No speech was detected."
    piper_bin = str(Path(sys.executable).parent / "piper")
    piper_cmd = [piper_bin, "-m", "assets/voices/en_US-lessac-medium.onnx", "--output-raw"]
    p = subprocess.Popen(piper_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_pcm, _ = p.communicate(input=f"{msg}\n".encode("utf-8"))
    
    pcm_audio = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    sd.play(pcm_audio, samplerate=22050, device="pulse")
    sd.wait()
    print("[TTS] Playback finished!")
except Exception as e:
    print(f"[TTS] Playback notice: {e}")

print("\nTest completed.")
