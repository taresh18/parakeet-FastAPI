from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class AudioChunkTranscriptionResponse(BaseModel):
    """Response model for audio chunk transcription endpoints."""
    text: str = Field(..., description="Transcribed text from audio chunk.")
    processing_time: float = Field(..., description="Time taken to process the audio in seconds.")
    audio_duration: float = Field(..., description="Duration of the audio chunk in seconds.") 