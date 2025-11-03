# face_manager.py
import os
import cv2
import json
import numpy as np
from pathlib import Path

DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
LABELS_FILE = MODELS_DIR / "labels.json"
MODEL_FILE = MODELS_DIR / "lbph_model.yml"

# Ensure directories exist
DATASET_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def get_face_detector():
    """Return a Haar Cascade face detector."""
    return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ... (rest of the code: save_face_image, gather_dataset, train_lbph_model, etc.)


def save_face_image(person_name: str, img, idx:int):
    person_dir = DATASET_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    path = person_dir / f"{person_name}_{idx:03d}.jpg"
    cv2.imwrite(str(path), img)
    return str(path)

def gather_dataset():
    """Return (images, labels, label_to_name) for training."""
    images = []
    labels = []
    label_to_name = {}
    name_to_label = {}
    current_label = 0

    for person_dir in sorted(DATASET_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        name = person_dir.name
        name_to_label[name] = current_label
        label_to_name[str(current_label)] = name

        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            images.append(img)
            labels.append(current_label)
        current_label += 1

    return images, labels, label_to_name

def train_lbph_model(min_samples=10):
    images, labels, label_to_name = gather_dataset()
    if len(images) == 0 or len(labels) < min_samples:
        raise ValueError(f"Not enough samples to train (found {len(labels)}). Capture more images per person.")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(images, np.array(labels))
    recognizer.write(str(MODEL_FILE))

    # save label mapping
    with open(LABELS_FILE, "w") as f:
        json.dump(label_to_name, f)
    return True

def load_model():
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        return None, {}
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))
    with open(LABELS_FILE, "r") as f:
        label_to_name = json.load(f)
    # keys are strings in JSON; convert to int-keyed dict for convenience
    return recognizer, {int(k): v for k, v in label_to_name.items()}

def list_registered_people():
    people = [p.name for p in DATASET_DIR.iterdir() if p.is_dir()]
    return sorted(people)
