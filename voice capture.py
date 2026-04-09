import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import os

# ─── SETTINGS ─────────────────────────────────────────────
SAMPLE_RATE = 16000
DURATION = 3        # seconds per recording
NUM_SAMPLES = 15    # number of samples per person

# ─── GET NAME ─────────────────────────────────────────────
name = input("Enter person's name: ").strip()
save_dir = f"voice_dataset/{name}"
os.makedirs(save_dir, exist_ok=True)

print(f"\n[INFO] Recording {NUM_SAMPLES} voice samples for: {name}")
print("[INFO] Say 'open the door' each time when prompted")
print("[INFO] Each recording is 3 seconds long\n")

# ─── RECORD SAMPLES ───────────────────────────────────────
for i in range(NUM_SAMPLES):
    input(f"Press ENTER when ready for sample {i+1}/{NUM_SAMPLES}...")
    print("🎤 Recording... speak now!")

    audio = sd.rec(int(DURATION * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE,
                   channels=1,
                   dtype='int16',
                   device=None)   # uses default mic
    sd.wait()

    path = f"{save_dir}/{i:03d}.wav"
    wav.write(path, SAMPLE_RATE, audio)
    print(f"[SAVED] {path}\n")

print(f"\n[DONE] Saved {NUM_SAMPLES} samples for {name}")
print(f"[INFO] Files saved in: {save_dir}/")
