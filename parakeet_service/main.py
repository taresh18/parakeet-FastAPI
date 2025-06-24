from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .model import lifespan
from .routes import router
from .config import logger

from parakeet_service.stream_routes import router as stream_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="Parakeet-TDT 0.6B v2 STT service",
        version="0.0.1",
        description=(
            "High-accuracy English speech-to-text (FastConformer-TDT) "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )
    
    # Add CORS middleware with optimizations
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    app.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)
    app.include_router(stream_router)
    
    logger.info("FastAPI app initialised with connection optimizations")
    return app


app = create_app()

# Add server configuration for connection optimization
if __name__ == "__main__":
    uvicorn.run(
        "parakeet_service.main:app",
        host="0.0.0.0",
        port=8989,
        # Connection optimization settings
        keepalive_timeout=300,  # 5 minutes keep-alive
        timeout_keep_alive=300,  # 5 minutes keep-alive
        limit_concurrency=100,  # Allow more concurrent connections
        limit_max_requests=10000,  # Allow more requests per connection
        # Performance optimizations
        workers=1,  # Single worker for GPU model
        loop="uvloop",  # Use uvloop for better performance
        http="httptools",  # Use httptools for better HTTP parsing
        access_log=False,  # Disable access logging for performance
    )
