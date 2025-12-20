from flask import Flask, request, jsonify
import joblib
import numpy as np
import cv2
from PIL import Image
import os
import io
import base64
import mediapipe as mp
import requests

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))

mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():

    # ===== INPUT HANDLING =====
    image = None

    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")

    elif request.is_json and "image" in request.json:
        image_bytes = base64.b64decode(request.json["image"])
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    elif request.is_json and "image_url" in request.json:
        resp = requests.get(request.json["image_url"], timeout=10)
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    else:
        return jsonify({"error": "No image provided"}), 400

    img = np.array(image)

    # ===== FACE DETECTION (RGB REQUIRED) =====
    results = mp_face.process(img)

    if not results.detections:
        return jsonify({"error": "No face detected"}), 400

    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape

    # ===== SAFE BOUNDING BOX =====
    x1 = max(0, int(bbox.xmin * w))
    y1 = max(0, int(bbox.ymin * h))
    x2 = min(w, x1 + int(bbox.width * w))
    y2 = min(h, y1 + int(bbox.height * h))

    face_img = img[y1:y2, x1:x2]

    if face_img.size == 0:
        return jsonify({"error": "Invalid face crop"}), 400

    # ===== PREPROCESS =====
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.flatten() / 255.0

    # ===== PREDICT =====
    probs = model.predict_proba([face])[0]

    result = {labels[i]: float(probs[i]) for i in range(len(labels))}
    dominant = labels[int(np.argmax(probs))]
    confidence = float(np.max(probs))

    return jsonify({
        "dominant_emotion": dominant,
        "confidence": confidence,
        "all_emotions": result
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
