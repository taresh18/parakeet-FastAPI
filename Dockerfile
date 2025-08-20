FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

RUN apt-get update && apt-get -y install libopenmpi-dev nano htop ffmpeg build-essential gcc g++ make cmake

RUN mkdir -p /workspace
COPY . /workspace/parakeet-FastAPI

WORKDIR /workspace/parakeet-FastAPI

RUN pip install numpy typing_extensions 
RUN pip install -r requirements.txt

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8989", "--limit-concurrency", "100", "--limit-max-requests", "10000", "--workers", "1", "--loop", "uvloop", "--http", "httptools", "--access-log", "--log-level", "warning"]


