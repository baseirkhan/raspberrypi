import os
import numpy as np
import pickle
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

os.makedirs("results", exist_ok=True)

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

# ─── LOAD VOICE DATASET ───────────────────────────────────
print("[INFO] Loading voice dataset...")
X_seq, X_flat, y_raw = [], [], []

for person in os.listdir("voice_dataset"):
    person_dir = f"voice_dataset/{person}"
    if not os.path.isdir(person_dir):
        continue
    for wav_file in os.listdir(person_dir):
        if not wav_file.endswith(".wav"):
            continue
        path = f"{person_dir}/{wav_file}"
        try:
            sr, audio = wav.read(path)
            audio = audio.astype(np.float32)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            mfcc = extract_mfcc(audio, sr)
            X_seq.append(mfcc)
            X_flat.append(np.mean(mfcc, axis=0))  # averaged for SVM/RF
            y_raw.append(person)
        except Exception as e:
            print(f"[WARN] Skipping {path}: {e}")

le = LabelEncoder()
y = le.fit_transform(y_raw)
label_names = list(le.classes_)

print(f"[INFO] People: {label_names}")
print(f"[INFO] Total samples: {len(y)}")

X_flat = np.array(X_flat)

# ─── METRICS HELPER ───────────────────────────────────────
def compute_metrics(y_true, y_pred, model_name, label_names):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*40}")
    print(f"  {model_name}")
    print(f"{'='*40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"results/cm_voice_{model_name.replace(' ', '_')}.png")
    plt.close()
    print(f"  Confusion matrix saved!")

    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}

results_list = []
X_train_f, X_test_f, y_train, y_test = train_test_split(
    X_flat, y, test_size=0.2, random_state=42, stratify=y)

# ─── MODEL 1: GMM ─────────────────────────────────────────
print("\n[INFO] Training GMM model...")
gmm_models = {}
for label in np.unique(y_train):
    name = label_names[label]
    person_mfcc = [X_seq[i] for i in range(len(y)) if y[i] == label]
    all_mfcc = np.vstack(person_mfcc)
    gmm = GaussianMixture(n_components=8, covariance_type='diag',
                          n_init=3, max_iter=200)
    gmm.fit(all_mfcc)
    gmm_models[label] = gmm

y_pred_gmm = []
test_indices = [i for i in range(len(y)) if i >= int(len(y) * 0.8)]
for idx in range(len(X_test_f)):
    mfcc_seq = X_seq[len(X_train_f) + idx]
    scores = {label: gmm_models[label].score(mfcc_seq)
              for label in gmm_models}
    y_pred_gmm.append(max(scores, key=scores.get))

r = compute_metrics(y_test, y_pred_gmm, "GMM", label_names)
results_list.append(r)

with open("results/gmm_voice_model.pickle", "wb") as f:
    pickle.dump(gmm_models, f)

# ─── MODEL 2: SVM ─────────────────────────────────────────
print("\n[INFO] Training SVM model...")
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
svm.fit(X_train_f, y_train)
y_pred_svm = svm.predict(X_test_f)

r = compute_metrics(y_test, y_pred_svm, "SVM", label_names)
results_list.append(r)

with open("results/svm_voice_model.pickle", "wb") as f:
    pickle.dump(svm, f)

# ─── MODEL 3: Random Forest ───────────────────────────────
print("\n[INFO] Training Random Forest model...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_f, y_train)
y_pred_rf = rf.predict(X_test_f)

r = compute_metrics(y_test, y_pred_rf, "Random Forest", label_names)
results_list.append(r)

with open("results/rf_voice_model.pickle", "wb") as f:
    pickle.dump(rf, f)

# ─── COMPARISON CHART ─────────────────────────────────────
print("\n[INFO] Generating comparison chart...")
metrics = ["accuracy", "precision", "recall", "f1"]
x = np.arange(len(metrics))
width = 0.25

fig, ax = plt.subplots(figsize=(12, 6))
for i, res in enumerate(results_list):
    vals = [res[m] for m in metrics]
    ax.bar(x + i * width, vals, width, label=res["model"])

ax.set_xlabel('Metric')
ax.set_ylabel('Score')
ax.set_title('Voice Recognition Model Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/voice_model_comparison.png")
plt.close()

print("\n[INFO] All results saved to results/ folder")
print("\n── FINAL SUMMARY ──")
for r in results_list:
    print(f"{r['model']:20s} | Acc: {r['accuracy']:.4f} | "
          f"Prec: {r['precision']:.4f} | "
          f"Rec: {r['recall']:.4f} | "
          f"F1: {r['f1']:.4f}")
