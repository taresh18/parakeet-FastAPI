#!/bin/bash

# Start the FastAPI application with uvicorn
exec uvicorn parakeet_service.main:app \
    --host 0.0.0.0 \
    --port 8989 \
    --timeout-keep-alive 300 \
    --log-level info