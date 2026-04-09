import os
import numpy as np
import scipy.io.wavfile as wav
import pickle
from sklearn.mixture import GaussianMixture

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

print("[INFO] Training voice models...")
voice_models = {}

for person in os.listdir("voice_dataset"):
    person_dir = f"voice_dataset/{person}"
    if not os.path.isdir(person_dir):
        continue

    print(f"[INFO] Processing {person}...")
    all_mfcc = []

    for wav_file in os.listdir(person_dir):
        if not wav_file.endswith(".wav"):
            continue
        path = f"{person_dir}/{wav_file}"
        try:
            sample_rate, audio = wav.read(path)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            mfcc = extract_mfcc(audio, sample_rate)
            all_mfcc.append(mfcc)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

    if len(all_mfcc) == 0:
        print(f"[WARN] No samples found for {person}")
        continue

    all_mfcc = np.vstack(all_mfcc)
    gmm = GaussianMixture(n_components=8, covariance_type='diag',
                          n_init=3, max_iter=200)
    gmm.fit(all_mfcc)
    voice_models[person] = gmm
    print(f"[INFO] Model trained for {person}")

with open("voice_encodings.pickle", "wb") as f:
    pickle.dump(voice_models, f)

print("[INFO] Voice training complete! Saved to voice_encodings.pickle")
