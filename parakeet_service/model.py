from contextlib import asynccontextmanager
import contextlib
import gc
import torch, asyncio
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecMultiTaskModel
from omegaconf import open_dict

from .config import MODEL_NAME, MODEL_PRECISION, DEVICE, logger

from parakeet_service.batchworker import batch_worker


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading ASR models with optimized memory...")
    with torch.inference_mode():
        # Determine precision
        dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32
        
        # Load Parakeet model with configurable device and precision
        logger.info(f"Loading Parakeet model: {MODEL_NAME}")
        parakeet_model = nemo_asr.models.ASRModel.from_pretrained(
            MODEL_NAME, 
            map_location=DEVICE
        ).to(dtype=dtype)
        logger.info("Loaded Parakeet model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)

        # Load Canary model
        logger.info("Loading Canary model: nvidia/canary-1b-flash")
        canary_model = EncDecMultiTaskModel.from_pretrained(
            'nvidia/canary-1b-flash',
            map_location=DEVICE
        ).to(dtype=dtype)
        # update decode params for canary
        with open_dict(canary_model.cfg.decoding):
            canary_model.cfg.decoding.beam.beam_size = 1
        canary_model.change_decoding_strategy(canary_model.cfg.decoding)
        logger.info("Loaded Canary model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)
        
    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    app.state.asr_model = parakeet_model
    app.state.canary_model = canary_model
    logger.info("Parakeet model ready on %s", next(parakeet_model.parameters()).device)
    logger.info("Canary model ready on %s", next(canary_model.parameters()).device)

    app.state.worker = asyncio.create_task(batch_worker(parakeet_model), name="batch_worker")
    logger.info("batch_worker scheduled for Parakeet model")

    try:
        yield
    finally:
        app.state.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.worker

        logger.info("Releasing GPU memory and shutting down worker")
                    
        del app.state.asr_model
        del app.state.canary_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    with open_dict(model.cfg.decoding):
        if getattr(model.cfg.decoding, "compute_timestamps", False):
            model.cfg.decoding.compute_timestamps = False
        if getattr(model.cfg.decoding, "preserve_alignments", False):
            model.cfg.decoding.preserve_alignments = False
    model.change_decoding_strategy(model.cfg.decoding)
