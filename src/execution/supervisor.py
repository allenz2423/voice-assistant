import asyncio
import os
import signal
import glob
from pathlib import Path

class HardenedJobSupervisor:
    """Spawns jobs in Bubblewrap sandboxes with full GPU access, path bindings, and bwrap parent death-signals."""
    def __init__(self, log_dir="~/.local/state/adam/jobs", workspace="~/workspace", downloads="~/Downloads"):
        self.log_dir = Path(log_dir).expanduser()
        self.workspace = Path(workspace).expanduser()
        self.downloads = Path(downloads).expanduser()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.downloads.mkdir(parents=True, exist_ok=True)
        self.active_jobs: dict[str, dict] = {}

    def _get_gpu_device_args(self) -> list[str]:
        args = []
        if os.path.exists("/dev/dri"):
            args.extend(["--dev-bind-try", "/dev/dri", "/dev/dri"])
        for dev_path in glob.glob("/dev/nvidia*"):
            args.extend(["--dev-bind-try", dev_path, dev_path])
        return args

    async def start_sandboxed_job(self, job_id: str, raw_cmd: list[str], arbiter=None) -> int:
        """Launches command inside an isolated Bubblewrap container with GPU access."""
        import re
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", job_id)
        log_file = (self.log_dir / f"{safe_id}.log").resolve()
        if not str(log_file).startswith(str(self.log_dir.resolve())):
            raise ValueError(f"Illegal job_id path traversal: {job_id}")
        out_fp = await asyncio.to_thread(open, log_file, "wb")

        # Canonical Arch Linux merged-/usr symlinks and isolated PID namespace
        bwrap_cmd = [
            "bwrap",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib", "/lib64",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/bin", "/sbin",
            "--ro-bind", "/etc", "/etc",
            "--ro-bind", "/sys", "/sys",                            # Needed for NVML / GPU topology
            "--proc", "/proc",
            "--dev", "/dev",
            "--tmpfs", "/dev/shm",                                  # CUDA/POSIX shared memory
            "--tmpfs", "/tmp",
            "--unshare-pid",                                        # bwrap acts as PID 1; auto-reaps all descendants
            "--ro-bind", str(self.downloads), str(self.downloads),  # Access downloads
            "--bind", str(self.workspace), str(self.workspace),      # Output directory
            "--chdir", str(self.workspace),
            "--die-with-parent",                                    # Kills sandbox if Adam exits
        ] + self._get_gpu_device_args() + ["--"] + raw_cmd

        proc = await asyncio.create_subprocess_exec(
            *bwrap_cmd,
            stdout=out_fp,
            stderr=asyncio.subprocess.STDOUT,
            stdin=asyncio.subprocess.DEVNULL,
            start_new_session=True
        )

        self.active_jobs[job_id] = {
            "proc": proc,
            "pgid": proc.pid,
            "out_fp": out_fp,
            "log_file": log_file
        }

        asyncio.create_task(self._watch_job(job_id, proc, out_fp, arbiter))
        return proc.pid

    async def kill_job(self, job_id: str):
        """Kills the process group cleanly."""
        job = self.active_jobs.get(job_id)
        if not job:
            return
        pgid = job["pgid"]
        try:
            os.killpg(pgid, signal.SIGTERM)
            await asyncio.sleep(1.0)
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    async def _watch_job(self, job_id: str, proc: asyncio.subprocess.Process, out_fp, arbiter):
        returncode = await proc.wait()
        await asyncio.to_thread(out_fp.close)
        self.active_jobs.pop(job_id, None)

        status = "succeeded" if returncode == 0 else f"failed with exit code {returncode}"
        print(f"[Supervisor] Background job '{job_id}' {status}.")
        if arbiter:
            await arbiter.enqueue_notification(
                priority=1,
                message=f"Background job {job_id} {status}."
            )
