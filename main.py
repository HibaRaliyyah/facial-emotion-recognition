from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import io
import base64

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    # INPUT
    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")
    elif request.is_json and "image" in request.json:
        image_bytes = base64.b64decode(request.json["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    else:
        return jsonify({"error": "No image provided"}), 400

    # PREPROCESS
    img = np.array(image)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48, 48)).flatten() / 255.0

    # PREDICT
    probs = model.predict_proba([face])[0]

    result = {labels[i]: float(probs[i]) for i in range(len(labels))}

    dominant = labels[int(np.argmax(probs))]

    return jsonify({
        "dominant_emotion": dominant,
        "confidence": float(np.max(probs)),
        "all_emotions": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
