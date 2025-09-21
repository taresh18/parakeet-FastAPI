import logging, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()   

# Configuration
PARAKEET_MODEL_NAME = os.getenv("PARAKEET_MODEL_NAME", "nvidia/parakeet-tdt-0.6b-v3")
CANARY_MODEL_NAME = os.getenv("CANARY_MODEL_NAME", "nvidia/canary-1b-flash")
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
BEAM_SIZE = int(os.getenv("BEAM_SIZE", "4"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "30"))
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "60"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_DIR = os.getenv("LOG_DIR", "logs")
LOG_FILE = "app.log"

# Global logger instance
_logger = None

def get_logger():
    global _logger
    
    if _logger is None:
        _logger = logging.getLogger("parakeet_service")
        _logger.setLevel(LOG_LEVEL)
        _logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s  %(levelname)-7s  %(name)s: %(message)s")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(LOG_LEVEL)
        console_handler.setFormatter(formatter)
        _logger.addHandler(console_handler)

        # File handler - ensure directory exists
        log_path = Path(LOG_DIR) / LOG_FILE
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

        _logger.propagate = False
    
    return _logger

logger = get_logger()
