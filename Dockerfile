# Training container for Vertex AI Custom Training Jobs.
#
# Local build/test:
#   docker build -t wine-mlops-train .
#   docker run --rm -v "$(pwd)/data:/app/data" wine-mlops-train
#   docker run --rm -v "$(pwd)/data:/app/data" wine-mlops-train --max-iter 2000
#
# Vertex AI passes hyperparameters as container args (see pipelines/ in Day 5),
# which is why ENTRYPOINT is fixed to train.py but args are left open via CMD.

FROM python:3.11-slim

WORKDIR /app

# Install deps first so this layer is cached across rebuilds unless
# requirements.txt itself changes (much faster iteration).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source last, since this changes most often.
COPY src/ ./src/

WORKDIR /app/src

ENV PYTHONUNBUFFERED=1

# ENTRYPOINT is fixed to train.py; CMD supplies default args that Vertex
# AI (or `docker run`) can override, e.g.:
#   docker run wine-mlops-train --max-iter 2000 --data-path gs://bucket/data.csv
ENTRYPOINT ["python", "train.py"]
CMD []