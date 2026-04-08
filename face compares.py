import os
import cv2
import numpy as np
import face_recognition
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from imutils import paths
import os

os.makedirs("results", exist_ok=True)

# ─── LOAD DATASET ─────────────────────────────────────────
print("[INFO] Loading face dataset...")
imagePaths = list(paths.list_images("dataset"))

hog_encodings, hog_labels = [], []
lbph_images,   lbph_labels = [], []
eigen_images,  eigen_labels = [], []

label_map = {}
label_counter = 0

IMG_SIZE = (100, 100)

for imagePath in imagePaths:
    name = imagePath.split(os.path.sep)[-2]

    # Assign numeric label
    if name not in label_map:
        label_map[name] = label_counter
        label_counter += 1
    label = label_map[name]

    image = cv2.imread(imagePath)
    if image is None:
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_resized = cv2.resize(gray, IMG_SIZE)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # HOG encoding
    boxes = face_recognition.face_locations(rgb, model="hog")
    encs = face_recognition.face_encodings(rgb, boxes)
    if len(encs) > 0:
        hog_encodings.append(encs[0])
        hog_labels.append(label)

    # LBPH and EigenFace use grayscale resized images
    lbph_images.append(gray_resized)
    lbph_labels.append(label)
    eigen_images.append(gray_resized)
    eigen_labels.append(label)

# Save label map
with open("results/label_map.pickle", "wb") as f:
    pickle.dump(label_map, f)

reverse_map = {v: k for k, v in label_map.items()}
print(f"[INFO] People: {list(label_map.keys())}")
print(f"[INFO] Total images: {len(lbph_images)}")

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

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f"results/cm_face_{model_name.replace(' ', '_')}.png")
    plt.close()
    print(f"  Confusion matrix saved!")

    return {"model": model_name, "accuracy": acc,
            "precision": prec, "recall": rec, "f1": f1}

results_list = []
label_names = [reverse_map[i] for i in range(len(label_map))]

# ─── MODEL 1: HOG + face_recognition ──────────────────────
print("\n[INFO] Training HOG model...")
if len(hog_encodings) > 10:
    X = np.array(hog_encodings)
    y = np.array(hog_labels)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Save train encodings
    train_data = {"encodings": list(X_train), "names": [reverse_map[l] for l in y_train]}
    with open("encodings.pickle", "wb") as f:
        pickle.dump(train_data, f)

    # Predict using nearest neighbor (face_recognition style)
    y_pred = []
    for enc in X_test:
        distances = face_recognition.face_distance(list(X_train), enc)
        best = np.argmin(distances)
        if distances[best] < 0.55:
            y_pred.append(y_train[best])
        else:
            y_pred.append(-1)  # Unknown

    # Filter unknowns for metrics
    valid = [(yt, yp) for yt, yp in zip(y_test, y_pred) if yp != -1]
    if valid:
        yt_v, yp_v = zip(*valid)
        r = compute_metrics(yt_v, yp_v, "HOG face_recognition", label_names)
        results_list.append(r)
else:
    print("[WARN] Not enough HOG samples, need more photos per person")

# ─── MODEL 2: LBPH ────────────────────────────────────────
print("\n[INFO] Training LBPH model...")
X = np.array(lbph_images)
y = np.array(lbph_labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

lbph = cv2.face.LBPHFaceRecognizer_create()
lbph.train(list(X_train), y_train)
lbph.save("results/lbph_model.yml")

y_pred = []
for img in X_test:
    label, confidence = lbph.predict(img)
    # confidence < 80 means good match in LBPH
    if confidence < 80:
        y_pred.append(label)
    else:
        y_pred.append(-1)

valid = [(yt, yp) for yt, yp in zip(y_test, y_pred) if yp != -1]
if valid:
    yt_v, yp_v = zip(*valid)
    r = compute_metrics(yt_v, yp_v, "LBPH", label_names)
    results_list.append(r)

# ─── MODEL 3: EigenFaces ──────────────────────────────────
print("\n[INFO] Training EigenFaces model...")
X = np.array(eigen_images)
y = np.array(eigen_labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

num_components = min(len(label_map) * 10, len(X_train) - 1)
eigen = cv2.face.EigenFaceRecognizer_create(num_components=num_components)
eigen.train(list(X_train), y_train)
eigen.save("results/eigen_model.yml")

y_pred = []
for img in X_test:
    label, confidence = eigen.predict(img)
    if confidence < 5000:
        y_pred.append(label)
    else:
        y_pred.append(-1)

valid = [(yt, yp) for yt, yp in zip(y_test, y_pred) if yp != -1]
if valid:
    yt_v, yp_v = zip(*valid)
    r = compute_metrics(yt_v, yp_v, "EigenFaces", label_names)
    results_list.append(r)

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
ax.set_title('Face Recognition Model Comparison')
ax.set_xticks(x + width)
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1 Score'])
ax.set_ylim(0, 1.1)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig("results/face_model_comparison.png")
plt.close()

print("\n[INFO] All results saved to results/ folder")
print("\n── FINAL SUMMARY ──")
for r in results_list:
    print(f"{r['model']:25s} | Acc: {r['accuracy']:.4f} | "
          f"Prec: {r['precision']:.4f} | "
          f"Rec: {r['recall']:.4f} | "
          f"F1: {r['f1']:.4f}")
