#!/bin/bash
export HF_HOME=/workspace/hf

uvicorn main:app \
    --host 0.0.0.0 \
    --port 8989 \
    --limit-concurrency 100 \
    --limit-max-requests 10000 \
    --workers 1 \
    --loop uvloop \
    --http httptools \
    --access-log \
    --log-level warning