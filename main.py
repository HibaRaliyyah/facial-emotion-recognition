from flask import Flask, request, jsonify
from flask_cors import CORS
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
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))

# ✅ CHANGE 1: Better MediaPipe config for selfies
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=1,          # was 0
    min_detection_confidence=0.3  # was 0.5
)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    image = None

    # 1️⃣ multipart/form-data (binary upload)
    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")

    # 2️⃣ image_url JSON (n8n sends THIS)
    elif request.is_json and "image_url" in request.json:
        resp = requests.get(request.json["image_url"], timeout=10)
        resp.raise_for_status()
        image = Image.open(io.BytesIO(resp.content)).convert("RGB")

    # 3️⃣ base64 JSON
    elif request.is_json and "image" in request.json:
        try:
            image_bytes = base64.b64decode(request.json["image"], validate=True)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception:
            return jsonify({"error": "Invalid base64 image"}), 400

    else:
        return jsonify({"error": "No image provided"}), 400

    # ===== IMAGE PROCESSING =====
    img = np.array(image)
    rgb = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = mp_face.process(img)

    if not results.detections:
        return jsonify({"error": "No face detected"}), 400

    bbox = results.detections[0].location_data.relative_bounding_box
    h, w, _ = img.shape
    
    x = int(bbox.xmin * w)
    y = int(bbox.ymin * h)
    bw = int(bbox.width * w)
    bh = int(bbox.height * h)

    # Add margin to match FER-2013 crops (which include more of the head)
    # MediaPipe bounding boxes are very tight around facial features
    margin_x = int(bw * 0.2)
    margin_y = int(bh * 0.3)  # More margin on top/bottom for forehead/chin
    
    x1 = max(0, x - margin_x)
    y1 = max(0, y - int(margin_y * 1.5)) # extra space for forehead
    x2 = min(w, x + bw + margin_x)
    y2 = min(h, y + bh + margin_y)

    face_img = img[y1:y2, x1:x2]
    
    # Handle case where crop is invalid (e.g., width or height is 0)
    if face_img.size == 0:
        return jsonify({"error": "Invalid face crop"}), 400
        
    gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
    face = cv2.resize(gray, (48, 48))
    face = face.flatten() / 255.0

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
