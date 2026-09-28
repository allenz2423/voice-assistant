import asyncio
import heapq
from enum import Enum

class SystemState(Enum):
    IDLE_LISTENING = "IDLE_LISTENING"
    USER_SPEAKING = "USER_SPEAKING"
    PROCESSING_REACT = "PROCESSING_REACT"
    ASSISTANT_SPEAKING = "ASSISTANT_SPEAKING"
    AWAITING_CONFIRMATION = "AWAITING_CONFIRMATION"

class PriorityAudioArbiter:
    """Manages audio access and notifications without conversational collision or lock deadlocks."""
    def __init__(self, tts_engine, earcon_engine):
        self.tts = tts_engine
        self.earcon = earcon_engine
        self.current_state = SystemState.IDLE_LISTENING
        self.notification_queue: list[tuple[int, str]] = []  # Min-heap (priority, message)
        self._state_lock = asyncio.Lock()
        self._drain_lock = asyncio.Lock()

    async def set_state(self, new_state: str):
        """Atomically transitions system state without blocking on audio playback."""
        should_drain = False
        async with self._state_lock:
            self.current_state = SystemState(new_state)
            if self.current_state == SystemState.IDLE_LISTENING and self.notification_queue:
                should_drain = True

        if should_drain:
            # Drain asynchronously in background without holding _state_lock
            asyncio.create_task(self._drain_notifications())

    async def enqueue_notification(self, priority: int, message: str):
        """Enqueues a background notification and plays if idle."""
        should_drain = False
        async with self._state_lock:
            heapq.heappush(self.notification_queue, (priority, message))
            if self.current_state == SystemState.IDLE_LISTENING:
                should_drain = True

        if should_drain:
            asyncio.create_task(self._drain_notifications())

    async def _drain_notifications(self):
        """Delivers pending notifications sequentially when idle, yielding locks between messages."""
        if self._drain_lock.locked():
            return

        async with self._drain_lock:
            while True:
                message = None
                async with self._state_lock:
                    if not self.notification_queue or self.current_state != SystemState.IDLE_LISTENING:
                        break
                    _, message = heapq.heappop(self.notification_queue)
                    self.current_state = SystemState.ASSISTANT_SPEAKING

                if message:
                    if self.earcon:
                        self.earcon.play("done")
                    await asyncio.sleep(0.05)
                    await self.tts.speak_async(message)

                    async with self._state_lock:
                        # Only return to IDLE_LISTENING if state wasn't changed by user interrupt during speech
                        if self.current_state == SystemState.ASSISTANT_SPEAKING:
                            self.current_state = SystemState.IDLE_LISTENING
