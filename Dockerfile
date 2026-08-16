FROM python:3.11-slim

WORKDIR /app

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ ./src/

WORKDIR /app/src

ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "train.py"]