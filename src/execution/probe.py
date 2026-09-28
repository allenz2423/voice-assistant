import asyncio
import os
import shutil

class HardwareEncoderProbe:
    """Probes GPU NVENC capabilities and driver compatibility, falling back to CPU SVT-AV1."""
    def __init__(self):
        self._cached_nvenc_works = None

    async def test_nvenc_allocation(self) -> bool:
        """Performs a 1-frame test encode to verify NVENC API version & device support."""
        if self._cached_nvenc_works is not None:
            return self._cached_nvenc_works

        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc=d=0.04:s=128x128",
            "-c:v", "hevc_nvenc", "-f", "null", "-"
        ]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL
            )
            returncode = await proc.wait()
            self._cached_nvenc_works = (returncode == 0)
        except Exception:
            self._cached_nvenc_works = False

        return self._cached_nvenc_works

    async def select_transcode_parameters(self, target_codec: str = "av1") -> dict:
        """Selects optimal encoder arguments based on hardware capabilities and driver support."""
        nvenc_available = await self.test_nvenc_allocation()

        codec_lower = target_codec.lower().strip()

        if "av1" in codec_lower:
            # RTX 3070 Ampere lacks hardware AV1 NVENC (requires RTX 40-series).
            # CPU SVT-AV1 runs at ~4x realtime on multi-core host.
            return {
                "encoder": "libsvtav1",
                "args": ["-c:v", "libsvtav1", "-preset", "8", "-crf", "28", "-c:a", "copy"],
                "reason": "RTX 3070 Ampere lacks hardware AV1 NVENC. Using high-performance CPU SVT-AV1 (~4x realtime)."
            }

        elif "hevc" in codec_lower or "265" in codec_lower:
            if nvenc_available:
                return {
                    "encoder": "hevc_nvenc",
                    "args": ["-c:v", "hevc_nvenc", "-preset", "p5", "-cq", "24", "-c:a", "copy"],
                    "reason": "Allocated hardware HEVC NVENC on GPU 1."
                }
            else:
                return {
                    "encoder": "libx265",
                    "args": ["-c:v", "libx265", "-preset", "fast", "-crf", "24", "-c:a", "copy"],
                    "reason": "NVIDIA driver 580 does not meet Arch FFmpeg 9.0.2 NVENC 13.1 API (requires driver >= 610). Soft-falling back to CPU libx265."
                }

        else:
            return {
                "encoder": "copy",
                "args": ["-c:v", "copy", "-c:a", "copy"],
                "reason": "Stream copy."
            }
