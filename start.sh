#!/bin/bash

# Start the FastAPI application with uvicorn
exec uvicorn parakeet_service.main:app \
    --host 0.0.0.0 \
    --port 8989 \
    --timeout-keep-alive 300 \
    # --limit-concurrency 100 \
    # --limit-max-requests 10000 \
    # --workers 1 \
    # --loop uvloop \
    # --http httptools \
    # --no-access-log \
    --log-level info