import face_recognition
import cv2
import numpy as np
import pickle
import time

print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

MATCH_THRESHOLD = 0.55
cv_scaler = 2
face_locations = []
face_names = []
frame_count = 0
start_time = time.time()
fps = 0

print("[INFO] press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0,0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb, model="hog")
    face_encodings = face_recognition.face_encodings(rgb, face_locations, model="small")

    face_names = []
    for enc in face_encodings:
        name = "Unknown"
        if len(known_face_encodings) > 0:
            distances = face_recognition.face_distance(known_face_encodings, enc)
            best = np.argmin(distances)
            if distances[best] < MATCH_THRESHOLD:
                name = known_face_names[best]
        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        top    *= cv_scaler
        right  *= cv_scaler
        bottom *= cv_scaler
        left   *= cv_scaler
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, top-30), (right, top), color, cv2.FILLED)
        cv2.putText(frame, name, (left+5, top-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
