# face_manager.py
from pathlib import Path
import cv2
from deepface import DeepFace

# Dataset where captured face images are stored
DATASET_DIR = Path("dataset")
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Folder where DeepFace stores its internal representations
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def save_face_image(person_name: str, face_image, index: int):
    """
    Save a single cropped face image for registration.
    """
    person_dir = DATASET_DIR / person_name
    person_dir.mkdir(parents=True, exist_ok=True)
    filename = person_dir / f"{person_name}_{index:03d}.jpg"
    cv2.imwrite(str(filename), face_image)
    print(f"[INFO] Saved {filename}")


def train_deepface_model():
    """
    DeepFace doesn't 'train' a model like LBPH does.
    Instead, it builds an internal embeddings index for fast lookup.

    We trigger it once so that DeepFace builds representations_facenet.pkl
    inside the dataset directory.
    """
    if not DATASET_DIR.exists() or not any(DATASET_DIR.rglob("*.jpg")):
        raise RuntimeError("No registered face images found in dataset/.")

    print("[INFO] Building DeepFace embeddings database...")

    # DeepFace.find() automatically builds or refreshes embeddings
    # for all faces in dataset/
    sample_image = next(DATASET_DIR.rglob("*.jpg"), None)
    DeepFace.find(
        img_path=str(sample_image),
        db_path=str(DATASET_DIR),
        model_name="Facenet",
        enforce_detection=False
    )

    print("[INFO] Face database built successfully.")


def list_registered_people():
    """
    Return all registered person names.
    """
    if not DATASET_DIR.exists():
        return []
    return sorted([d.name for d in DATASET_DIR.iterdir() if d.is_dir()])
