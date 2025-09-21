import torch
import torchaudio.functional as AF
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecMultiTaskModel
from omegaconf import open_dict
import gc
import time

from .utils import PARAKEET_MODEL_NAME, CANARY_MODEL_NAME, DEVICE, logger, BATCH_SIZE


class CanaryModel:
    """Canary ASR model wrapper with optimized memory management."""
    
    def __init__(self):
        """Initialize Canary model"""
        logger.info("Loading Canary model...")
        with torch.inference_mode():
            dtype = torch.float16
            logger.info(f"Loading Canary model: {CANARY_MODEL_NAME}")
            self.model = EncDecMultiTaskModel.from_pretrained(
                CANARY_MODEL_NAME,
                map_location=DEVICE
            ).to(dtype=dtype)
            # update decode params for canary
            with open_dict(self.model.cfg.decoding):
                self.model.cfg.decoding.beam.beam_size = 1
            self.model.change_decoding_strategy(self.model.cfg.decoding)
            logger.info(f"Loaded Canary model with fp16 weights on {DEVICE}")
        
        # Aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Canary model ready on %s", next(self.model.parameters()).device)
    
    def infer(self, audio_data: bytes, sample_rate: int) -> dict:
        """
        Perform inference on raw audio data with punctuation and capitalization.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate of the audio data
            
        Returns:
            dict: Contains 'text', 'processing_time', 'audio_duration'
        """
        start_time = time.perf_counter()
        
        try:
            audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
            audio_duration = audio_samples / sample_rate
            
            # Zero-copy conversion to torch tensor, then normalize to -1.0 to 1.0
            # Move tensor to GPU immediately if available
            device = next(self.model.parameters()).device
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
                results = self.model.transcribe(
                    audio=audio_array,
                    batch_size=BATCH_SIZE,
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
            
            processing_time = time.perf_counter() - start_time
            
            logger.debug(f"Canary transcription completed in {processing_time:.3f}s: '{transcribed_text}'")
            
            return {
                'text': transcribed_text,
                'processing_time': processing_time,
                'audio_duration': audio_duration
            }
            
        except Exception as exc:
            processing_time = time.perf_counter() - start_time
            logger.exception(f"Canary transcription failed after {processing_time:.3f}s")
            raise RuntimeError(f"Canary transcription failed: {str(exc)}") from exc
    
    def cleanup(self):
        """Clean up model resources."""
        logger.info("Releasing Canary GPU memory")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class ParakeetModel:
    """Parakeet ASR model wrapper with optimized memory management."""
    
    def __init__(self):
        """Initialize Parakeet model"""
        logger.info("Loading Parakeet model...")
        with torch.inference_mode():
            dtype = torch.float16
            logger.info(f"Loading Parakeet model: {PARAKEET_MODEL_NAME}")
            self.model = nemo_asr.models.ASRModel.from_pretrained(
                PARAKEET_MODEL_NAME,
                map_location=DEVICE
            ).to(dtype=dtype)
            logger.info(f"Loaded Parakeet model with fp16 weights on {DEVICE}")
        
        # Aggressive cleanup
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Parakeet model ready on %s", next(self.model.parameters()).device)
    
    def infer(self, audio_data: bytes, sample_rate: int) -> dict:
        """
        Perform inference on raw audio data.
        
        Args:
            audio_data: Raw audio bytes (16-bit PCM)
            sample_rate: Sample rate of the audio data
            
        Returns:
            dict: Contains 'text', 'processing_time', 'audio_duration'
        """
        start_time = time.perf_counter()
        
        try:
            audio_samples = len(audio_data) // 2  # 16-bit = 2 bytes per sample
            audio_duration = audio_samples / sample_rate
            
            device = next(self.model.parameters()).device
            audio_tensor = torch.frombuffer(audio_data, dtype=torch.int16).float() / 32768.0
            if device.type == 'cuda':
                audio_tensor = audio_tensor.to(device)
            
            # Only resample if necessary
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
                results = self.model.transcribe(
                    audio=[audio_array],
                    batch_size=BATCH_SIZE,
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
            
            processing_time = time.perf_counter() - start_time
            
            logger.debug(f"Parakeet transcription completed in {processing_time:.3f}s: '{transcribed_text}'")
            
            return {
                'text': transcribed_text,
                'processing_time': processing_time,
                'audio_duration': audio_duration
            }
            
        except Exception as exc:
            processing_time = time.perf_counter() - start_time
            logger.exception(f"Parakeet transcription failed after {processing_time:.3f}s")
            raise RuntimeError(f"Parakeet transcription failed: {str(exc)}") from exc
    
    def cleanup(self):
        """Clean up model resources."""
        logger.info("Releasing Parakeet GPU memory")
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 