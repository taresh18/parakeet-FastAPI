import asyncio
import time

import numpy as np
import torch
import torchaudio.functional as AF

from .utils import logger


def preprocess_audio(
    audio_data: bytes, sample_rate: int, device: torch.device
) -> tuple[np.ndarray, float]:
    """
    Convert raw 16-bit PCM bytes to a numpy array at 16 kHz.

    Returns:
        (audio_array, audio_duration) - numpy float32 array and duration in seconds.
    """
    audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
    audio_duration = audio_samples / sample_rate

    audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0

    if device.type == "cuda":
        audio_tensor = audio_tensor.to(device)

    if sample_rate != 16000:
        audio_tensor = audio_tensor.unsqueeze(0)
        audio_tensor = AF.resample(audio_tensor, sample_rate, 16000)
        audio_tensor = audio_tensor.squeeze(0)

    if device.type == "cuda":
        audio_array = audio_tensor.cpu().numpy()
    else:
        audio_array = audio_tensor.numpy()

    return audio_array, audio_duration


class BatchInferenceWorker:
    """
    Async batch-inference worker for NeMo ASR models.

    Collects concurrent requests via an asyncio.Queue, groups them into
    batches (bounded by *max_batch_size* and *max_wait_ms*), then runs a
    single ``model.transcribe()`` call per batch in a thread-pool executor.

    This guarantees thread-safe model access (only one transcribe call at a
    time) while maximising GPU utilisation through batching.
    """

    def __init__(
        self,
        model,
        model_type: str,
        max_batch_size: int = 8,
        max_wait_ms: float = 50.0,
    ):
        """
        Args:
            model: The underlying NeMo ASR model (e.g. ``ParakeetModel.model``).
            model_type: ``"parakeet"`` or ``"canary"`` – controls transcribe kwargs.
            max_batch_size: Maximum number of requests per batch.
            max_wait_ms: Maximum milliseconds to wait after the first request
                         arrives before dispatching a (possibly partial) batch.
        """
        self.model = model
        self.model_type = model_type
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # store as seconds
        self._queue: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the background batch-processing loop."""
        loop = asyncio.get_event_loop()
        self._task = loop.create_task(self._run())
        logger.info(
            "%s batch worker started  (max_batch=%d, max_wait=%.0fms)",
            self.model_type,
            self.max_batch_size,
            self.max_wait_ms * 1000,
        )

    def stop(self) -> None:
        """Cancel the background loop."""
        if self._task is not None:
            self._task.cancel()
            self._task = None

    # ------------------------------------------------------------------
    # Public API – called by endpoint handlers
    # ------------------------------------------------------------------

    async def submit(self, audio_array: np.ndarray, audio_duration: float) -> dict:
        """
        Enqueue a preprocessed audio array for transcription.

        Blocks (awaits) until the batch containing this request has been
        processed by the background worker.

        Returns:
            dict with ``text``, ``processing_time``, ``audio_duration``.
        """
        future = asyncio.get_event_loop().create_future()
        await self._queue.put(
            {
                "audio_array": audio_array,
                "audio_duration": audio_duration,
                "start_time": time.perf_counter(),
                "future": future,
            }
        )
        return await future

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    async def _run(self) -> None:
        loop = asyncio.get_event_loop()

        while True:
            try:
                # Block until the first request arrives.
                item = await self._queue.get()
                batch = [item]

                # Drain up to max_batch_size-1 more items within the time window.
                deadline = loop.time() + self.max_wait_ms
                while len(batch) < self.max_batch_size:
                    remaining = deadline - loop.time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(
                            self._queue.get(), timeout=remaining
                        )
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                batch_size = len(batch)
                logger.debug(
                    "%s batch dispatching %d request(s)", self.model_type, batch_size
                )

                # Sort by audio length to minimise padding waste.
                batch.sort(key=lambda x: len(x["audio_array"]))

                arrays = [item["audio_array"] for item in batch]
                futures = [item["future"] for item in batch]
                durations = [item["audio_duration"] for item in batch]
                start_times = [item["start_time"] for item in batch]

                try:
                    texts = await loop.run_in_executor(
                        None, self._transcribe_batch, arrays
                    )
                    now = time.perf_counter()
                    for i, fut in enumerate(futures):
                        if not fut.done():
                            fut.set_result(
                                {
                                    "text": texts[i],
                                    "processing_time": now - start_times[i],
                                    "audio_duration": durations[i],
                                }
                            )
                except Exception as exc:
                    logger.exception(
                        "%s batch transcription failed", self.model_type
                    )
                    for fut in futures:
                        if not fut.done():
                            fut.set_exception(exc)

            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("%s batch worker unexpected error", self.model_type)

    # ------------------------------------------------------------------
    # Blocking transcription – runs inside the thread-pool executor
    # ------------------------------------------------------------------

    def _transcribe_batch(self, arrays: list[np.ndarray]) -> list[str]:
        """Call ``model.transcribe()`` on a list of numpy arrays."""
        with torch.inference_mode():
            kwargs: dict = {
                "audio": arrays,
                "batch_size": len(arrays),
            }
            if self.model_type == "parakeet":
                kwargs["timestamps"] = False
            elif self.model_type == "canary":
                kwargs["pnc"] = "yes"

            results = self.model.transcribe(**kwargs)

        # NeMo may return (hypotheses, _) tuple.
        if isinstance(results, tuple):
            results = results[0]

        texts: list[str] = []
        if isinstance(results, list):
            for r in results:
                texts.append(getattr(r, "text", str(r)))
        else:
            texts.append(str(results) if results else "")

        return texts
