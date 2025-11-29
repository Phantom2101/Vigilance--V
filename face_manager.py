# face_manager.py
import os
import cv2
import numpy as np
from pathlib import Path
import pickle

# Folder where all face images will be stored (dataset/person_name/*.jpg)
DATASET_DIR = Path("dataset")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODELS_DIR / "lbph_model.yml"
LABELS_FILE = MODELS_DIR / "labels.pkl"


def get_face_detector():
    """Return an OpenCV Haar cascade face detector."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def save_face_image(person_name: str, face_image, index: int):
    """
    Save one face image (already cropped and resized) to that person's dataset folder.
    """
    person_dir = DATASET_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    file_path = person_dir / f"{person_name}_{index:03d}.jpg"
    cv2.imwrite(str(file_path), face_image)


def list_registered_people():
    """Return a list of registered person names (folder names in dataset)."""
    if not DATASET_DIR.exists():
        return []
    return sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])


def train_lbph_model():
    """
    Train an LBPH face recognizer from the dataset directory.
    Each subfolder of dataset/ is treated as a person's images.
    """
    print("[TRAIN] Loading dataset for LBPH training...")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces = []
    labels = []
    label_map = {}
    current_label = 0

    if not DATASET_DIR.exists():
        raise RuntimeError("Dataset folder does not exist. Please register faces first.")

    for person_dir in DATASET_DIR.iterdir():
        if not person_dir.is_dir():
            continue
        label_map[current_label] = person_dir.name
        print(f"[TRAIN] Reading images for {person_dir.name}")
        for img_path in person_dir.glob("*.jpg"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [WARN] Cannot read {img_path}")
                continue
            faces.append(img)
            labels.append(current_label)
        current_label += 1

    if len(faces) == 0:
        raise RuntimeError("No face images found for training.")

    recognizer.train(faces, np.array(labels))
    recognizer.save(str(MODEL_FILE))

    # Save label map for decoding predictions
    with open(LABELS_FILE, "wb") as f:
        pickle.dump(label_map, f)

    print(f"[TRAIN] Model trained and saved to {MODEL_FILE}")
    print(f"[TRAIN] Labels saved to {LABELS_FILE}")


def load_model():
    """
    Load a trained LBPH model and label map if available.
    Returns (recognizer, label_map)
    """
    if not MODEL_FILE.exists() or not LABELS_FILE.exists():
        print("[INFO] No trained model found. Run training first.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(str(MODEL_FILE))

    with open(LABELS_FILE, "rb") as f:
        label_map = pickle.load(f)

    return recognizer, label_map
