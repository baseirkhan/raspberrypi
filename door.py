import cv2
import face_recognition
import numpy as np
import pickle
import time
import threading
import speech_recognition as sr
import RPi.GPIO as GPIO

# ─── GPIO SETUP ───────────────────────────────────────────
RED_LED   = 16   # door closed
GREEN_LED = 26   # door open

GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED,   GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)

# Start with red ON (door closed)
GPIO.output(RED_LED,   True)
GPIO.output(GREEN_LED, False)

# ─── LOAD FACE ENCODINGS ──────────────────────────────────
print("[INFO] loading encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names = data["names"]

# ─── WEBCAM ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    GPIO.cleanup()
    exit()

# ─── SETTINGS ─────────────────────────────────────────────
MATCH_THRESHOLD = 0.55
cv_scaler = 2

# ─── SHARED STATE ─────────────────────────────────────────
face_verified = False
voice_verified = False
led_on = False

# ─── LED CONTROL ──────────────────────────────────────────
def open_door():
    global led_on, face_verified, voice_verified
    print("[DOOR] Access granted! Opening door...")

    # Red OFF, Green ON
    GPIO.output(RED_LED,   False)
    GPIO.output(GREEN_LED, True)
    led_on = True

    time.sleep(5)  # door open for 5 seconds

    # Green OFF, Red ON (door closed again)
    GPIO.output(GREEN_LED, False)
    GPIO.output(RED_LED,   True)
    led_on = False
    face_verified = False
    voice_verified = False
    print("[DOOR] Door closed. System reset.")

def blink_green(times=3):
    """Blink green to show face recognized, waiting for voice"""
    for _ in range(times):
        GPIO.output(GREEN_LED, True)
        time.sleep(0.2)
        GPIO.output(GREEN_LED, False)
        time.sleep(0.2)

# ─── VOICE RECOGNITION ────────────────────────────────────
recognizer = sr.Recognizer()
mic = sr.Microphone(device_index=1)  # webcam mic

TRIGGER_PHRASES = ["open the door", "open door", "unlock", "open"]

def listen_for_command():
    global voice_verified, face_verified
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        while True:
            try:
                print("[VOICE] Listening...")
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=4)
                text = recognizer.recognize_google(audio).lower()
                print(f"[VOICE] Heard: {text}")

                if any(phrase in text for phrase in TRIGGER_PHRASES):
                    print("[VOICE] Trigger phrase detected!")
                    voice_verified = True

                    if face_verified and not led_on:
                        threading.Thread(target=open_door, daemon=True).start()

            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                print("[VOICE] Could not understand audio")
            except Exception as e:
                print(f"[VOICE ERROR] {e}")

# Start voice thread
voice_thread = threading.Thread(target=listen_for_command, daemon=True)
voice_thread.start()

# ─── MAIN LOOP ────────────────────────────────────────────
print("[INFO] System ready. Red LED ON = door closed.")
print("[INFO] Face + voice required to open door.")
print("[INFO] Press 'q' to quit")

frame_count = 0
start_time  = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Downscale for faster processing
    small = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    # Detect faces using HOG
    face_locations    = face_recognition.face_locations(rgb, model="hog")
    face_encodings_list = face_recognition.face_encodings(rgb, face_locations, model="small")

    current_face_detected = False

    for (top, right, bottom, left), enc in zip(face_locations, face_encodings_list):
        x1 = left   * cv_scaler
        y1 = top    * cv_scaler
        x2 = right  * cv_scaler
        y2 = bottom * cv_scaler

        name = "Unknown"
        if len(known_face_encodings) > 0:
            distances  = face_recognition.face_distance(known_face_encodings, enc)
            best_index = np.argmin(distances)
            if distances[best_index] < MATCH_THRESHOLD:
                name = known_face_names[best_index]
                current_face_detected = True

                if not face_verified:
                    face_verified = True
                    print(f"[FACE] Recognized: {name} - Say 'open the door'")
                    threading.Thread(target=blink_green, daemon=True).start()

                    if voice_verified and not led_on:
                        threading.Thread(target=open_door, daemon=True).start()

        # Draw box and name
        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, name, (x1+5, y1-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    if not current_face_detected:
        face_verified = False

    # ── Status display ─────────────────────────────────────
    face_status = "✓ Face OK"   if face_verified else "✗ No Face"
    voice_status= "✓ Voice OK"  if voice_verified else "✗ No Voice"
    door_status = "🟢 DOOR OPEN" if led_on        else "🔴 DOOR CLOSED"

    cv2.putText(frame, face_status,  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if face_verified else (0,0,255), 2)
    cv2.putText(frame, voice_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,0) if voice_verified else (0,0,255), 2)
    cv2.putText(frame, door_status,  (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0,255,255) if led_on else (0,0,255), 2)

    # FPS
    frame_count += 1
    elapsed = time.time() - start_time
    if elapsed > 1:
        fps = frame_count / elapsed
        frame_count = 0
        start_time = time.time()
    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    cv2.imshow("Door Access System", frame)
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
print("[INFO] System stopped")
