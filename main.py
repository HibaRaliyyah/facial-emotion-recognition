from flask import Flask, request, jsonify
import numpy as np
import cv2
import mediapipe as mp
import joblib
from PIL import Image
import io
import base64

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

# Load model
model = joblib.load("emotion_model.joblib")
labels = joblib.load("labels.joblib")

# MediaPipe
mp_face = mp.solutions.face_detection
face_detector = mp_face.FaceDetection(
    model_selection=1,
    min_detection_confidence=0.5
)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/predict", methods=["POST"])
def predict():

    # ---------- INPUT ----------
    if "image" in request.files:
        image = Image.open(request.files["image"].stream).convert("RGB")

    elif request.is_json:
        b64 = request.json.get("image")
        image_bytes = base64.b64decode(b64)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    else:
        return jsonify({"error": "No image provided"}), 400

    img = np.array(image)
    h, w, _ = img.shape

    # ---------- FACE DETECTION ----------
    results = face_detector.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    if not results.detections:
        return jsonify({"error": "No face detected"}), 400

    bbox = results.detections[0].location_data.relative_bounding_box
    x, y = int(bbox.xmin * w), int(bbox.ymin * h)
    bw, bh = int(bbox.width * w), int(bbox.height * h)

    face = cv2.cvtColor(img[y:y+bh, x:x+bw], cv2.COLOR_RGB2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.flatten().reshape(1, -1) / 255.0

    # ---------- PREDICT ----------
    probs = model.predict_proba(face)[0]
    result = {labels[i]: float(probs[i]) for i in range(len(labels))}

    return jsonify({
        "dominant_emotion": labels[int(np.argmax(probs))],
        "confidence": float(np.max(probs)),
        "all_emotions": result
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
