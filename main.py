from __future__ import annotations
import time
from contextlib import asynccontextmanager
import contextlib

from fastapi import FastAPI, HTTPException, Query, status, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Import configuration and models
from src.utils import logger
from src.schemas import AudioChunkTranscriptionResponse
from src.models import ParakeetModel, CanaryModel


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading ASR models with optimized memory...")
    
    # Initialize model instances
    parakeet_model = ParakeetModel()
    canary_model = CanaryModel()
    
    app.state.parakeet_model = parakeet_model
    app.state.canary_model = canary_model
    logger.info("Models loaded and ready")

    try:
        yield
    finally:
        logger.info("Releasing GPU memory")
        parakeet_model.cleanup()
        canary_model.cleanup()

# Create FastAPI app
app = FastAPI(
    title="Parakeet-TDT 0.6B v2 STT service",
    version="0.0.1",
    description=(
        "High-accuracy English speech-to-text with parakeet and canary models "
    ),
    lifespan=lifespan,
)

# Endpoints
@app.post(
    "/v1/transcribe/parakeet",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe raw audio data using Parakeet model",
    description="Transcribe raw audio data (16-bit PCM, mono, any sample rate) via request body using Parakeet model"
)
async def transcribe_raw_audio_chunk(
    request: Request,
    sample_rate: int = Query(..., description="Sample rate of the audio data")
):
    """
    Transcribe raw audio data via request body.
    """
    try:
        model = request.app.state.parakeet_model
        
        # Read raw audio data from request body
        audio_data = await request.body()
        
        # Use the model's infer method
        result = model.infer(audio_data, sample_rate)
        
        return AudioChunkTranscriptionResponse(
            text=result['text'],
            processing_time=result['processing_time'],
            audio_duration=result['audio_duration']
        )
        
    except Exception as exc:
        logger.exception("Raw audio transcription failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Raw audio transcription failed: {str(exc)}"
        ) from exc


@app.post(
    "/v1/transcribe/canary",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe raw audio data using Canary model",
    description="Transcribe raw audio data (16-bit PCM, mono, any sample rate) directly via request body using Canary model"
)
async def transcribe_raw_audio_chunk_canary(
    request: Request,
    sample_rate: int = Query(..., description="Sample rate of the audio data")
):
    """
    Transcribe raw audio data via request body with Canary model.
    """
    try:
        model = request.app.state.canary_model
        
        # Read raw audio data from request body
        audio_data = await request.body()
        
        # Use the model's infer method
        result = model.infer(audio_data, sample_rate)
        
        return AudioChunkTranscriptionResponse(
            text=result['text'],
            processing_time=result['processing_time'],
            audio_duration=result['audio_duration']
        )
        
    except Exception as exc:
        logger.exception("Canary raw audio transcription failed")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Canary raw audio transcription failed: {str(exc)}"
        ) from exc


logger.info("FastAPI app initialised with connection optimizations")