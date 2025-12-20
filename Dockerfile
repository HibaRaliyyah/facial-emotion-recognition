FROM python:3.10-slim

WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# create models directory
RUN mkdir -p models

# ⬇️ Download models from Hugging Face
RUN curl -L -o models/emotion_model.joblib \
    https://huggingface.co/hibaraliyyah/emotion-recognition-sklearn/resolve/main/emotion_model.joblib

RUN curl -L -o models/labels.joblib \
    https://huggingface.co/hibaraliyyah/emotion-recognition-sklearn/resolve/main/labels.joblib

COPY . .

CMD ["gunicorn", "-b", "0.0.0.0:8000", "main:app"]
