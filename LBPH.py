import cv2
import numpy as np
import os
import pickle
from imutils import paths

# ─── SETTINGS ─────────────────────────────────────────────
IMG_SIZE = (200, 200)
CONFIDENCE_THRESHOLD = 80  # lower = stricter match

# ─── LOAD DATASET ─────────────────────────────────────────
print("[INFO] Loading dataset...")
imagePaths = list(paths.list_images("dataset"))

faces = []
labels = []
label_map = {}
label_counter = 0

for imagePath in imagePaths:
    name = imagePath.split(os.path.sep)[-2]

    # Assign numeric label
    if name not in label_map:
        label_map[name] = label_counter
        label_counter += 1
    label = label_map[name]

    # Load and convert to grayscale
    image = cv2.imread(imagePath)
    if image is None:
        continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, IMG_SIZE)

    faces.append(gray)
    labels.append(label)

print(f"[INFO] People found: {list(label_map.keys())}")
print(f"[INFO] Total images: {len(faces)}")

# Save label map
with open("lbph_labels.pickle", "wb") as f:
    pickle.dump(label_map, f)

# ─── TRAIN LBPH MODEL ─────────────────────────────────────
print("[INFO] Training LBPH model...")
lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(faces, np.array(labels))
lbph.save("lbph_model.yml")
print("[INFO] Model saved to lbph_model.yml")

# ─── REAL TIME RECOGNITION ────────────────────────────────
print("[INFO] Starting webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Reverse label map for lookup
reverse_map = {v: k for k, v in label_map.items()}

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

import time
frame_count = 0
start_time = time.time()
fps = 0

print("[INFO] Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    detected_faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )

    for (x, y, w, h) in detected_faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, IMG_SIZE)

        # Predict
        label, confidence = lbph.predict(face_roi)

        if confidence < CONFIDENCE_THRESHOLD:
            name = reverse_map.get(label, "Unknown")
            color = (0, 200, 0)  # green
        else:
            name = "Unknown"
            color = (0, 0, 220)  # red

        # Display confidence score
        display = f"{name} ({confidence:.1f})"

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.rectangle(frame, (x, y-30), (x+w, y), color, cv2.FILLED)
        cv2.putText(frame, display, (x+5, y-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    # FPS counter
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()

    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, "Model: LBPH", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("LBPH Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print("[INFO] Done")
