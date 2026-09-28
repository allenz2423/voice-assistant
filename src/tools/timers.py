import time
import asyncio
import uuid
from typing import Optional

class TimerManager:
    """Manages concurrent non-blocking async timers with audio alerts."""
    def __init__(self, tts_engine=None, earcon_engine=None):
        self.tts = tts_engine
        self.earcon = earcon_engine
        self.timers: dict[str, dict] = {}

    def set_timer(self, duration_seconds: int, label: Optional[str] = "timer") -> str:
        """Schedules a non-blocking countdown timer."""
        sec = max(1, int(duration_seconds))
        timer_id = str(uuid.uuid4())[:8]
        lbl = (label or "timer").strip()
        end_time = time.time() + sec

        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._timer_worker(timer_id, sec, lbl))
        except RuntimeError:
            task = None

        self.timers[timer_id] = {
            "id": timer_id,
            "label": lbl,
            "duration": sec,
            "end_time": end_time,
            "task": task
        }

        mins = sec // 60
        rem_sec = sec % 60
        time_desc = f"{mins} minutes" if mins > 0 and rem_sec == 0 else (f"{sec} seconds" if mins == 0 else f"{mins} minutes and {rem_sec} seconds")
        return f"Timer set for {time_desc} for {lbl}."

    async def _timer_worker(self, timer_id: str, duration: int, label: str):
        try:
            await asyncio.sleep(duration)
            if timer_id in self.timers:
                del self.timers[timer_id]

            # Play audible alert
            if self.earcon:
                self.earcon.play("done")
                await asyncio.sleep(0.2)
                self.earcon.play("done")

            # Speak alert
            mins = duration // 60
            dur_str = f"{mins} minute" if mins == 1 else (f"{mins} minutes" if mins > 1 else f"{duration} seconds")
            alert_msg = f"Your {dur_str} timer for {label} is done."
            if self.tts:
                await self.tts.speak_async(alert_msg)
        except asyncio.CancelledError:
            pass

    def list_timers(self) -> str:
        """Lists active countdown timers."""
        if not self.timers:
            return "There are no active timers."

        now = time.time()
        lines = []
        for tid, tinfo in list(self.timers.items()):
            remaining = max(0, int(tinfo["end_time"] - now))
            m = remaining // 60
            s = remaining % 60
            lines.append(f"{tinfo['label']} (ID {tid}): {m}m {s}s remaining")

        return "Active timers: " + ", ".join(lines)

    def cancel_timer(self, timer_id_or_label: str) -> str:
        """Cancels an active timer by ID or matching label."""
        target = (timer_id_or_label or "").strip().lower()
        if not target:
            return "Please specify a timer ID or label to cancel."

        to_cancel = []
        for tid, tinfo in list(self.timers.items()):
            if target == tid.lower() or target in tinfo["label"].lower():
                to_cancel.append((tid, tinfo))

        if not to_cancel:
            return f"No active timer found matching '{timer_id_or_label}'."

        for tid, tinfo in to_cancel:
            if tinfo.get("task"):
                tinfo["task"].cancel()
            del self.timers[tid]

        return f"Cancelled {len(to_cancel)} timer(s)."
