import cv2
import face_recognition
import numpy as np
import pickle
import time
import threading
import sounddevice as sd
import scipy.io.wavfile as wav
import RPi.GPIO as GPIO

# ─── GPIO SETUP ───────────────────────────────────────────
RED_LED   = 16
GREEN_LED = 26
GPIO.setmode(GPIO.BCM)
GPIO.setup(RED_LED,   GPIO.OUT)
GPIO.setup(GREEN_LED, GPIO.OUT)
GPIO.output(RED_LED,   True)   # Red ON by default
GPIO.output(GREEN_LED, False)

# ─── LOAD FACE ENCODINGS ──────────────────────────────────
print("[INFO] loading face encodings...")
with open("encodings.pickle", "rb") as f:
    data = pickle.loads(f.read())
known_face_encodings = data["encodings"]
known_face_names     = data["names"]

# ─── LOAD VOICE MODELS ────────────────────────────────────
print("[INFO] loading voice models...")
with open("voice_encodings.pickle", "rb") as f:
    voice_models = pickle.load(f)
print(f"[INFO] Voice models loaded for: {list(voice_models.keys())}")

# ─── WEBCAM ───────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("[ERROR] Cannot open webcam")
    GPIO.cleanup()
    exit()

# ─── SETTINGS ─────────────────────────────────────────────
FACE_THRESHOLD  = 0.55
VOICE_THRESHOLD = -50    # lower = stricter
SAMPLE_RATE     = 16000
DURATION        = 3      # seconds to record voice
cv_scaler       = 2

# ─── STATE ────────────────────────────────────────────────
face_verified      = False
face_verified_name = None
voice_verified      = False
voice_verified_name = None
led_on = False

# ─── MFCC EXTRACTION ──────────────────────────────────────
def extract_mfcc(audio, sample_rate, num_mfcc=13):
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    frame_length = int(round(0.025 * sample_rate))
    frame_step   = int(round(0.010 * sample_rate))
    signal_length = len(emphasized)
    num_frames = 1 + (signal_length - frame_length) // frame_step
    indices = (np.tile(np.arange(0, frame_length), (num_frames, 1)) +
               np.tile(np.arange(0, num_frames * frame_step, frame_step),
                       (frame_length, 1)).T)
    frames = emphasized[indices.astype(np.int32)]
    frames *= np.hamming(frame_length)
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0 / NFFT) * (mag_frames ** 2)
    num_filters = 26
    low_mel  = 0
    high_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_pts  = np.linspace(low_mel, high_mel, num_filters + 2)
    hz_pts   = 700 * (10 ** (mel_pts / 2595) - 1)
    bin_pts  = np.floor((NFFT + 1) * hz_pts / sample_rate).astype(int)
    fbank = np.zeros((num_filters, int(np.floor(NFFT / 2 + 1))))
    for m in range(1, num_filters + 1):
        for k in range(bin_pts[m-1], bin_pts[m]):
            fbank[m-1, k] = (k - bin_pts[m-1]) / (bin_pts[m] - bin_pts[m-1])
        for k in range(bin_pts[m], bin_pts[m+1]):
            fbank[m-1, k] = (bin_pts[m+1] - k) / (bin_pts[m+1] - bin_pts[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    mfcc = np.zeros((num_frames, num_mfcc))
    for n in range(num_mfcc):
        mfcc[:, n] = np.sum(filter_banks *
                            np.cos(np.pi * n / num_filters *
                                   (np.arange(1, num_filters + 1) - 0.5)), axis=1)
    return mfcc

def identify_voice(audio):
    """Returns name of recognized speaker or None"""
    try:
        mfcc = extract_mfcc(audio.astype(np.float32), SAMPLE_RATE)
        best_score = float('-inf')
        best_name  = None
        for name, gmm in voice_models.items():
            score = gmm.score(mfcc)
            print(f"[VOICE] {name} score: {score:.2f}")
            if score > best_score:
                best_score = score
                best_name  = name
        if best_score > VOICE_THRESHOLD:
            return best_name
        return None
    except Exception as e:
        print(f"[VOICE ERROR] {e}")
        return None

# ─── LED CONTROL ──────────────────────────────────────────
def open_door():
    global led_on, face_verified, voice_verified
    global face_verified_name, voice_verified_name
    print("[DOOR] Access granted! Opening door...")
    led_on = True
    GPIO.output(RED_LED,   False)
    GPIO.output(GREEN_LED, True)
    time.sleep(5)
    GPIO.output(GREEN_LED, False)
    GPIO.output(RED_LED,   True)
    led_on = False
    face_verified       = False
    voice_verified      = False
    face_verified_name  = None
    voice_verified_name = None
    print("[DOOR] Door closed. System reset.")

def blink_green(times=3):
    for _ in range(times):
        GPIO.output(GREEN_LED, True)
        time.sleep(0.2)
        GPIO.output(GREEN_LED, False)
        time.sleep(0.2)

# ─── VOICE LISTENER THREAD ────────────────────────────────
def listen_for_voice():
    global voice_verified, voice_verified_name
    global face_verified, face_verified_name, led_on

    while True:
        try:
            print("[VOICE] Listening... speak now!")
            audio = sd.rec(int(DURATION * SAMPLE_RATE),
                           samplerate=SAMPLE_RATE,
                           channels=1,
                           dtype='int16',
                           device=1)   # webcam mic
            sd.wait()
            audio = audio.flatten()

            name = identify_voice(audio)

            if name:
                print(f"[VOICE] Recognized voice: {name}")
                voice_verified      = True
                voice_verified_name = name

                # Check if SAME person verified by face
                if (face_verified and
                    face_verified_name == voice_verified_name and
                    not led_on):
                    threading.Thread(target=open_door, daemon=True).start()
            else:
                print("[VOICE] Unknown voice or score too low")
                voice_verified      = False
                voice_verified_name = None

        except Exception as e:
            print(f"[VOICE ERROR] {e}")

threading.Thread(target=listen_for_voice, daemon=True).start()

# ─── MAIN LOOP ────────────────────────────────────────────
print("[INFO] System ready!")
print("[INFO] Same person must be verified by BOTH face and voice")
print("[INFO] Press 'q' to quit")

frame_count = 0
start_time  = time.time()
fps = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small = cv2.resize(frame, (0, 0), fx=1/cv_scaler, fy=1/cv_scaler)
    rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    face_locations      = face_recognition.face_locations(rgb, model="hog")
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
            if distances[best_index] < FACE_THRESHOLD:
                name = known_face_names[best_index]
                current_face_detected = True

                if not face_verified or face_verified_name != name:
                    face_verified      = True
                    face_verified_name = name
                    print(f"[FACE] Recognized: {name} - Now speak!")
                    threading.Thread(target=blink_green, daemon=True).start()

                    if (voice_verified and
                        voice_verified_name == name and
                        not led_on):
                        threading.Thread(target=open_door, daemon=True).start()

        color = (0, 200, 0) if name != "Unknown" else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y1-30), (x2, y1), color, cv2.FILLED)
        cv2.putText(frame, name, (x1+5, y1-8),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255), 1)

    if not current_face_detected:
        face_verified      = False
        face_verified_name = None

    # Status
    face_status  = f"Face:  {face_verified_name}"  if face_verified  else "Face:  Not Detected"
    voice_status = f"Voice: {voice_verified_name}" if voice_verified else "Voice: Not Verified"
    door_status  = "DOOR: OPEN"   if led_on else "DOOR: CLOSED"

    cv2.putText(frame, face_status,  (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0,255,0) if face_verified else (0,0,255), 2)
    cv2.putText(frame, voice_status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0,255,0) if voice_verified else (0,0,255), 2)
    cv2.putText(frame, door_status,  (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0,255,255) if led_on else (0,0,255), 2)

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
