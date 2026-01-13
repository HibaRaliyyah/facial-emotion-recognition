import os

# =====================================================
# 🔥 FORCE CPU MODE (MUST BE FIRST — BEFORE ANY IMPORTS)
# =====================================================
os.environ["MEDIAPIPE_DISABLE_GPU"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from flask import Flask, request, jsonify
import io
import base64
import numpy as np
import cv2
from PIL import Image
import joblib

# 🚫 DO NOT use: import mediapipe as mp
# ✅ CPU-SAFE import
from mediapipe.python.solutions import face_detection

# =====================================================
# APP SETUP
# =====================================================
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# =====================================================
# GLOBAL SINGLETONS (LOW MEMORY)
# =====================================================
model = None
labels = None
detector = None


# =====================================================
# LOAD MODEL (LAZY)
# =====================================================
def load_model():
    global model, labels
    if model is None:
        print("🔹 Loading emotion model...")
        model = joblib.load(os.path.join(MODEL_DIR, "emotion_model.joblib"))
        labels = joblib.load(os.path.join(MODEL_DIR, "labels.joblib"))
    return model, labels


# =====================================================
# LOAD FACE DETECTOR (CPU ONLY)
# =====================================================
def load_face_detector():
    global detector
    if detector is None:
        print("🔹 Loading MediaPipe FaceDetection (CPU ONLY)")
        detector = face_detection.FaceDetection(
            model_selection=0,            # CPU SAFE
            min_detection_confidence=0.3
        )
    return detector


# =====================================================
# HEALTH CHECK
# =====================================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# =====================================================
# EMOTION PREDICTION
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        image_b64 = request.json.get("image")
        if not image_b64:
            return jsonify({"error": "No image provided"}), 400

        # =======================
        # DECODE BASE64 IMAGE
        # =======================
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Resize improves detection & reduces memory
        image.thumbnail((640, 640))
        img = np.array(image)

        # MediaPipe expects RGB
        rgb = img

        # =======================
        # FACE DETECTION
        # =======================
        detector = load_face_detector()
        results = detector.process(rgb)

        if not results.detections:
            return jsonify({
                "error": "No face detected",
                "hint": "Use a clear frontal face image with good lighting"
            }), 400

        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box

        h, w, _ = img.shape
        x1 = max(0, int(bbox.xmin * w))
        y1 = max(0, int(bbox.ymin * h))
        x2 = min(w, int((bbox.xmin + bbox.width) * w))
        y2 = min(h, int((bbox.ymin + bbox.height) * h))

        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            return jsonify({"error": "Face crop failed"}), 400

        # =======================
        # PREPROCESS FOR MODEL
        # =======================
        gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
        face = cv2.resize(gray, (48, 48))
        face = face.flatten() / 255.0

        # =======================
        # PREDICTION
        # =======================
        model, labels = load_model()
        probs = model.predict_proba([face])[0]

        emotions = {labels[i]: float(probs[i]) for i in range(len(labels))}

        return jsonify({
            "dominant_emotion": labels[int(np.argmax(probs))],
            "confidence": float(np.max(probs)),
            "all_emotions": emotions
        })

    except Exception as e:
        print("🔥 Prediction error:", str(e))
        return jsonify({"error": "Prediction failed"}), 500


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)
