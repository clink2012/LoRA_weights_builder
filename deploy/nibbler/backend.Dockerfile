FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/Database/backend

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY Database/backend/requirements.txt ./requirements.txt
COPY Database/backend/requirements-docker.txt ./requirements-docker.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-docker.txt

COPY Database/backend/ ./

EXPOSE 5001

CMD ["uvicorn", "lora_api_server_docker:app", "--host", "0.0.0.0", "--port", "5001"]
