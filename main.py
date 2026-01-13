from flask import Flask, request, jsonify
import os
import io
import base64
import numpy as np
import cv2
from PIL import Image
import joblib
import requests
import mediapipe as mp

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ================================
# 🔥 GLOBAL SINGLETONS (LAZY LOAD)
# ================================
model = None
labels = None
mp_face = None


def load_model():
    global model, labels
    if model is None:
        print("🔹 Loading ML model...")
        model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
        labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))
    return model, labels


def load_face_detector():
    global mp_face
    if mp_face is None:
        print("🔹 Loading MediaPipe Face Detector...")
        mp_face = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.3
        )
    return mp_face


# ================================
# ROUTES
# ================================
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image = None

        # 1️⃣ multipart/form-data
        if "image" in request.files:
            image = Image.open(request.files["image"].stream).convert("RGB")

        # 2️⃣ image_url
        elif request.is_json and "image_url" in request.json:
            resp = requests.get(request.json["image_url"], timeout=10)
            resp.raise_for_status()
            image = Image.open(io.BytesIO(resp.content)).convert("RGB")

        # 3️⃣ base64 JSON
        elif request.is_json and "image" in request.json:
            image_bytes = base64.b64decode(request.json["image"])
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        else:
            return jsonify({"error": "No image provided"}), 400

        img = np.array(image)
        rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mp_face = load_face_detector()
        results = mp_face.process(rgb)

        if not results.detections:
            return jsonify({"error": "No face detected"}), 400

        bbox = results.detections[0].location_data.relative_bounding_box
        h, w, _ = img.shape

        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            return jsonify({"error": "Face crop failed"}), 400

        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.flatten() / 255.0

        model, labels = load_model()
        probs = model.predict_proba([face])[0]

        result = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return jsonify({
            "dominant_emotion": labels[int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
            "all_emotions": result
        })

    except Exception as e:
        print("🔥 Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
