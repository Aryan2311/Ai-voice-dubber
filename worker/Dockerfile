# GPU worker. Build from repo root: docker build -f worker/Dockerfile .
# PyTorch >= 2.4 required by Coqui TTS (XTTS v2)
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

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
# StyleTTS2 + NLTK 3.8+ need punkt_tab at inference (not only punkt)
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

COPY worker/ /app/worker/

ENV PYTHONPATH=/app
# Run as module so "worker" is the package (worker/), not the script file (worker.py)
CMD ["python", "-m", "worker.worker"]
