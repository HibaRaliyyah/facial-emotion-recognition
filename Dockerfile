FROM python:3.10-slim

WORKDIR /app

# System dependencies for OpenCV & MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy models + app
COPY . .

# Fly.io listens on 8080
EXPOSE 8080

CMD ["gunicorn", "-b", "0.0.0.0:8080", "main:app"]
