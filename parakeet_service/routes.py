from __future__ import annotations
import time
import logging
import torch
import torchaudio.functional as AF

from fastapi import APIRouter, HTTPException, Query, status, Request, Form, File

from .schemas import AudioChunkTranscriptionResponse
from .config import logger


router = APIRouter(tags=["speech"])


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/v1/transcribe",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe audio chunk data directly",
    description="Transcribe raw audio chunk data (16-bit PCM, mono, any sample rate) directly without file upload"
)
async def transcribe_audio_chunk(
    request: Request,
    audio_data: bytes = File(..., description="Raw audio data (16-bit PCM, mono)"),
    sample_rate: int = Form(..., description="Sample rate of the audio data")
):
    """
    Transcribe audio chunk data directly.
    Optimized for maximum speed.
    
    Audio format requirements:
    - 16-bit PCM format
    - Mono channel
    - Any sample rate (will be resampled to 16kHz internally)
    """
    start_time = time.time()
    
    try:
        model = request.app.state.asr_model
        
        audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
        audio_duration = audio_samples / sample_rate
        
        logger.debug(f"Processing audio chunk: {len(audio_data)} bytes, {audio_samples} samples, "
                   f"{audio_duration:.3f}s duration")
        
        # Zero-copy conversion to torch tensor, then normalize to -1.0 to 1.0
        # Move tensor to GPU immediately if available
        device = next(model.parameters()).device
        audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
        if device.type == 'cuda':
            audio_tensor = audio_tensor.to(device)
        
        # Only resample if necessary using torch (on GPU if available)
        if sample_rate != 16000:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_tensor = AF.resample(audio_tensor, sample_rate, 16000)
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
            logger.debug(f"Resampling {sample_rate}Hz -> 16kHz on {device}")
        else:
            logger.debug("No resampling needed (already 16kHz)")
        
        # Convert back to CPU numpy for model inference (if model expects numpy)
        if device.type == 'cuda':
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor.numpy()
        
        with torch.inference_mode():
            results = model.transcribe(
                audio=audio_array,
                batch_size=1,
                timestamps=False,
            )
        logger.debug("Direct numpy array transcription successful")

        
        # Extract text
        if isinstance(results, tuple):
            results = results[0]
            
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            transcribed_text = getattr(result, 'text', str(result))
        else:
            transcribed_text = str(results) if results else ""
        
        processing_time = time.time() - start_time
        
        logger.debug(f"Transcription completed in {processing_time:.3f}s: '{transcribed_text}'")
        
        return AudioChunkTranscriptionResponse(
            text=transcribed_text,
            processing_time=processing_time,
            audio_duration=audio_duration
        )
        
    except Exception as exc:
        processing_time = time.time() - start_time
        logger.exception(f"Audio chunk transcription failed after {processing_time:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Audio chunk transcription failed: {str(exc)}"
        ) from exc


@router.post(
    "/v1/transcribe-raw",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe raw audio data directly",
    description="Transcribe raw audio data (16-bit PCM, mono, any sample rate) directly via request body - maximum speed"
)
async def transcribe_raw_audio_chunk(
    request: Request,
    sample_rate: int = Query(..., description="Sample rate of the audio data")
):
    """
    Transcribe raw audio data directly via request body.
    Ultra-optimized for maximum speed - no multipart parsing overhead.
    
    Audio format requirements:
    - 16-bit PCM format
    - Mono channel
    - Any sample rate (will be resampled to 16kHz internally)
    - Send as raw bytes in request body with Content-Type: application/octet-stream
    """
    start_time = time.time()
    
    try:
        model = request.app.state.asr_model
        
        # Read raw audio data from request body
        audio_data = await request.body()
        
        audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
        audio_duration = audio_samples / sample_rate
        
        # Zero-copy conversion to torch tensor, then normalize to -1.0 to 1.0
        # Move tensor to GPU immediately if available
        device = next(model.parameters()).device
        audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
        if device.type == 'cuda':
            audio_tensor = audio_tensor.to(device)
        
        # Only resample if necessary using torch (on GPU if available)
        if sample_rate != 16000:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_tensor = AF.resample(audio_tensor, sample_rate, 16000)
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
        
        # Convert back to CPU numpy for model inference (if model expects numpy)
        if device.type == 'cuda':
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor.numpy()
        
        with torch.inference_mode():
            results = model.transcribe(
                audio=audio_array,
                batch_size=1,
                timestamps=False,
            )
        
        # Fast text extraction
        if isinstance(results, tuple):
            results = results[0]
            
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            transcribed_text = getattr(result, 'text', str(result))
        else:
            transcribed_text = str(results) if results else ""
        
        processing_time = time.time() - start_time
        
        logger.debug(f"Raw transcription completed in {processing_time:.3f}s: '{transcribed_text}'")
        
        return AudioChunkTranscriptionResponse(
            text=transcribed_text,
            processing_time=processing_time,
            audio_duration=audio_duration
        )
        
    except Exception as exc:
        processing_time = time.time() - start_time
        logger.exception(f"Raw audio transcription failed after {processing_time:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Raw audio transcription failed: {str(exc)}"
        ) from exc


@router.post(
    "/v1/transcribe-raw/canary",
    response_model=AudioChunkTranscriptionResponse,
    summary="Transcribe raw audio data directly using Canary model",
    description="Transcribe raw audio data (16-bit PCM, mono, any sample rate) directly via request body using Canary model"
)
async def transcribe_raw_audio_chunk_canary(
    request: Request,
    sample_rate: int = Query(..., description="Sample rate of the audio data")
):
    """
    Transcribe raw audio data directly via request body with Canary model.
    Ultra-optimized for maximum speed - no multipart parsing overhead.
    Adds punctuation and capitalization.
    
    Audio format requirements:
    - 16-bit PCM format
    - Mono channel
    - Any sample rate (will be resampled to 16kHz internally)
    - Send as raw bytes in request body with Content-Type: application/octet-stream
    """
    start_time = time.time()
    
    try:
        model = request.app.state.canary_model
        
        # Read raw audio data from request body
        audio_data = await request.body()
        
        audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
        audio_duration = audio_samples / sample_rate
        
        # Zero-copy conversion to torch tensor, then normalize to -1.0 to 1.0
        # Move tensor to GPU immediately if available
        device = next(model.parameters()).device
        audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
        if device.type == 'cuda':
            audio_tensor = audio_tensor.to(device)
        
        # Only resample if necessary using torch (on GPU if available)
        if sample_rate != 16000:
            audio_tensor = audio_tensor.unsqueeze(0)  # Add batch dimension
            audio_tensor = AF.resample(audio_tensor, sample_rate, 16000)
            audio_tensor = audio_tensor.squeeze(0)  # Remove batch dimension
        
        # Convert to numpy for model inference
        if device.type == 'cuda':
            audio_array = audio_tensor.cpu().numpy()
        else:
            audio_array = audio_tensor.numpy()
        
        with torch.inference_mode():
            results = model.transcribe(
                audio=audio_array,
                batch_size=1,
                pnc='yes',  # Punctuation and Capitalization
            )
        
        # Fast text extraction
        if isinstance(results, tuple):
            results = results[0]
            
        if isinstance(results, list) and len(results) > 0:
            result = results[0]
            transcribed_text = getattr(result, 'text', str(result))
        else:
            transcribed_text = str(results) if results else ""
        
        processing_time = time.time() - start_time
        
        logger.debug(f"Canary raw transcription completed in {processing_time:.3f}s: '{transcribed_text}'")
        
        return AudioChunkTranscriptionResponse(
            text=transcribed_text,
            processing_time=processing_time,
            audio_duration=audio_duration
        )
        
    except Exception as exc:
        processing_time = time.time() - start_time
        logger.exception(f"Canary raw audio transcription failed after {processing_time:.3f}s")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Canary raw audio transcription failed: {str(exc)}"
        ) from exc


@router.get("/debug/cfg")
def show_cfg(request: Request):
    from omegaconf import OmegaConf
    model = request.app.state.asr_model         
    yaml_str = OmegaConf.to_yaml(model.cfg, resolve=True) 
    return yaml_str