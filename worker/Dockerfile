# GPU worker. Build from repo root: docker build -f worker/Dockerfile .
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# apt update + ffmpeg (and deps) is a large download; only ffmpeg needed for worker
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# force-confdef/confold = use defaults for any config questions (no prompt)
RUN apt-get update && apt-get install -y -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold" \
    --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY worker/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY worker/ /app/worker/

ENV PYTHONPATH=/app
CMD ["python", "worker/worker.py"]
