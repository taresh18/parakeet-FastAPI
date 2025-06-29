from pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

COPY . /root/apps/parakeet-fastapi

WORKDIR /root/apps/parakeet-fastapi

RUN pip install -r requirements.txt

EXPOSE 8989

CMD ["uvicorn", "parakeet_service.main:app", "--host", "0.0.0.0", "--port", "8989", "--timeout-keep-alive", "300", "--log-level", "info"]


# mkdir -p /root/apps/ && cd /root/apps && git clone https://github.com/taresh18/parakeet-FastAPI.git && cd parakeet-FastAPI && pip install -r requirements.txt && chmod +x start.sh && bash start.sh