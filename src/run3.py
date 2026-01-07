"""
Run 3: Dense SIFT + BoVW + Spatial Pyramid + One-vs-All Linear SVM

Core idea
- Extract Dense SIFT descriptors on a regular grid
- Learn a visual vocabulary using KMeans on training data only
- Encode each image as a BoVW histogram with Spatial Pyramid pooling
  1x1 + 2x2 levels capture both content and coarse layout
- Apply power normalization then L2 normalization
- Train a one-vs-all Linear SVM
"""

# =====================================================================
# Imports
# =====================================================================

import numpy as np
from pathlib import Path

import cv2

from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsRestClassifier


# =====================================================================
# Paths and configuration
# =====================================================================

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

VALID_EXTS = (".jpg", ".jpeg", ".png")

# Dense SIFT settings
SIFT_STEP = 6
SIFT_SIZE = 8

# Vocabulary size
VOCAB_SIZE = 600

# How many descriptors to sample from each image when building vocabulary
MAX_DESC_PER_IMAGE = 250

# Spatial pyramid levels
# Level 0 is 1x1, Level 1 is 2x2
PYRAMID_LEVELS = [1, 2]

# Random seed for repeatability
RANDOM_STATE = 0


# =====================================================================
# Dataset loading
# =====================================================================

def load_training_dataset(folder):
    """
    Loads training images and labels from a folder.

    Expected directory structure
        training/
            class_name/
                1.jpg
                2.jpg

    :param folder: Path to training dataset
    :return image_paths: List of image paths
    :return labels: List of class labels
    """
    folder = Path(folder)
    image_paths = []
    labels = []

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        label = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            image_paths.append(img_path)
            labels.append(label)

    return image_paths, np.array(labels)


def load_test_dataset(folder):
    """
    Loads test images only.

    Filenames are kept so the output matches the submission format.

    :param folder: Path to test dataset
    :return image_paths: List of image paths
    :return filenames: List of filenames
    """
    folder = Path(folder)

    sorted_img_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS],
        key=lambda p: int(p.stem)
    )

    image_paths = list(sorted_img_paths)
    filenames = [p.name for p in sorted_img_paths]

    return image_paths, filenames


def read_gray(path):
    """
    Loads an image in grayscale float32.

    :param path: Path to image
    :return img: Grayscale image array
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return img.astype(np.float32)


# =====================================================================
# Dense SIFT feature extraction
# =====================================================================

def dense_sift(img, step=SIFT_STEP, size=SIFT_SIZE):
    """
    Extracts Dense SIFT descriptors on a regular grid.

    We use OpenCV SIFT from opencv-contrib-python.

    :param img: Grayscale image float32
    :param step: Grid stride in pixels
    :param size: Keypoint size for SIFT
    :return desc: N x 128 array of SIFT descriptors
    :return coords: N x 2 array of keypoint coordinates (x, y)
    """
    h, w = img.shape

    # SIFT expects uint8 input
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)

    sift = cv2.SIFT_create()

    keypoints = []
    coords = []

    # Dense grid of keypoints
    for y in range(size, h - size, step):
        for x in range(size, w - size, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), float(size)))
            coords.append([x, y])

    if len(keypoints) == 0:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    keypoints, desc = sift.compute(img_u8, keypoints)

    if desc is None:
        return np.zeros((0, 128), dtype=np.float32), np.zeros((0, 2), dtype=np.int32)

    return desc.astype(np.float32), np.array(coords, dtype=np.int32)


# =====================================================================
# Vocabulary learning
# =====================================================================

def build_vocab(train_paths, vocab_size=VOCAB_SIZE, max_desc_per_image=MAX_DESC_PER_IMAGE):
    """
    Builds a visual vocabulary using MiniBatchKMeans.

    We sample a fixed number of descriptors per image for speed and balance.

    :param train_paths: List of training image paths
    :param vocab_size: Number of clusters
    :param max_desc_per_image: Max descriptors to keep per image
    :return kmeans: Trained MiniBatchKMeans
    """
    all_desc = []

    for path in train_paths:
        img = read_gray(path)
        desc, _ = dense_sift(img)

        if desc.shape[0] == 0:
            continue

        # Randomly sample descriptors from this image
        if desc.shape[0] > max_desc_per_image:
            idx = np.random.choice(desc.shape[0], size=max_desc_per_image, replace=False)
            desc = desc[idx]

        all_desc.append(desc)

    if len(all_desc) == 0:
        raise RuntimeError("No SIFT descriptors found to build a vocabulary")

    all_desc = np.vstack(all_desc)

    kmeans = MiniBatchKMeans(
        n_clusters=vocab_size,
        batch_size=4096,
        random_state=RANDOM_STATE,
        n_init=3
    )
    kmeans.fit(all_desc)

    return kmeans


# =====================================================================
# BoVW with Spatial Pyramid encoding
# =====================================================================

def power_l2_normalize(x, alpha=0.5, eps=1e-8):
    """
    Applies power normalization then L2 normalization.

    This is very important for histogram-like features.
    It reduces burstiness and improves linear separability.

    :param x: Feature vector
    :param alpha: Power exponent, 0.5 is common
    :param eps: Numerical stability constant
    :return x_norm: Normalized feature vector
    """
    x = np.sign(x) * (np.abs(x) ** alpha)
    n = np.linalg.norm(x)
    return x / (n + eps)


def bovw_spm(img, kmeans, vocab_size=VOCAB_SIZE, pyramid_levels=PYRAMID_LEVELS):
    """
    Encodes an image using BoVW with Spatial Pyramid pooling.

    Level 1x1 gives global content
    Level 2x2 adds coarse layout which helps scenes a lot

    Steps
    - Extract dense SIFT descriptors and their coordinates
    - Assign each descriptor to nearest visual word
    - For each pyramid level, build histograms per spatial bin
    - Concatenate all histograms
    - Apply power and L2 normalization

    :param img: Grayscale image float32
    :param kmeans: Trained vocabulary model
    :param vocab_size: Vocabulary size
    :param pyramid_levels: List of grid sizes, for example [1, 2]
    :return feat: Final feature vector
    """
    desc, coords = dense_sift(img)

    if desc.shape[0] == 0:
        # Return a valid zero vector if no descriptors exist
        total_bins = sum(l * l for l in pyramid_levels)
        feat = np.zeros(total_bins * vocab_size, dtype=np.float32)
        return feat

    words = kmeans.predict(desc)

    h, w = img.shape
    feats = []

    for l in pyramid_levels:
        # Bin sizes in pixels
        bin_w = w / l
        bin_h = h / l

        for by in range(l):
            for bx in range(l):
                x0 = bx * bin_w
                x1 = (bx + 1) * bin_w
                y0 = by * bin_h
                y1 = (by + 1) * bin_h

                # Select descriptors whose keypoints fall inside this spatial bin
                in_bin = (
                    (coords[:, 0] >= x0) & (coords[:, 0] < x1) &
                    (coords[:, 1] >= y0) & (coords[:, 1] < y1)
                )

                bin_words = words[in_bin]

                hist = np.zeros(vocab_size, dtype=np.float32)
                if bin_words.size > 0:
                    hist += np.bincount(bin_words, minlength=vocab_size).astype(np.float32)

                feats.append(hist)

    feat = np.concatenate(feats, axis=0)

    # Normalize for better performance
    feat = power_l2_normalize(feat)

    return feat.astype(np.float32)


def extract_features(paths, kmeans):
    """
    Extracts BoVW+SPM features for a list of image paths.

    :param paths: List of image paths
    :param kmeans: Trained vocabulary model
    :return X: Feature matrix
    """
    X = []
    for p in paths:
        img = read_gray(p)
        X.append(bovw_spm(img, kmeans))
    return np.array(X, dtype=np.float32)


# =====================================================================
# Training and validation
# =====================================================================

def train_and_validate_ovr_svm(X_train, y_train, X_val, y_val):
    """
    Trains and evaluates a one-vs-all Linear SVM with a small C sweep.

    :param X_train: Training features
    :param y_train: Training labels
    :param X_val: Validation features
    :param y_val: Validation labels
    :return best_clf: Trained classifier with best C
    :return best_C: Best C
    :return best_acc: Best validation accuracy
    """
    C_values = [0.3, 1.0, 3.0, 10.0]
    best_acc = -1.0
    best_C = C_values[0]
    best_clf = None

    for C in C_values:
        clf = OneVsRestClassifier(
            LinearSVC(C=C, class_weight="balanced", max_iter=15000),
            n_jobs=None
        )

        clf.fit(X_train, y_train)
        pred = clf.predict(X_val)
        acc = accuracy_score(y_val, pred)

        print(f"C={C}, validation accuracy={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            best_C = C
            best_clf = clf

    return best_clf, best_C, best_acc


# =====================================================================
# Main run
# =====================================================================

def run_bovw_sift_spm(train_dir, test_dir, run_number=3):
    """
    Full Run 3 pipeline.

    Steps
    - Load training paths and labels
    - Split into training and validation
    - Build vocabulary on training split only
    - Encode train and val using BoVW+SPM
    - Tune C on validation
    - Rebuild vocabulary on full training set
    - Retrain final classifier on full training set
    - Predict test labels and write run3.txt

    :param train_dir: Training folder name
    :param test_dir: Testing folder name
    :param run_number: Output run number
    """
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    all_paths, all_labels = load_training_dataset(train_path)

    # Encode labels consistently
    le = LabelEncoder()
    all_labels_enc = le.fit_transform(all_labels)

    # Split paths and labels
    train_paths, val_paths, y_train, y_val = train_test_split(
        all_paths,
        all_labels_enc,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=all_labels_enc
    )

    print("Building vocabulary on training split...")
    kmeans = build_vocab(train_paths, vocab_size=VOCAB_SIZE)

    print("Extracting BoVW+SPM features for training split...")
    X_train = extract_features(train_paths, kmeans)

    print("Extracting BoVW+SPM features for validation split...")
    X_val = extract_features(val_paths, kmeans)

    print("Training classifier (validation phase)...")
    _, best_C, val_acc = train_and_validate_ovr_svm(X_train, y_train, X_val, y_val)
    print(f"Best C: {best_C}")
    print(f"Validation accuracy: {val_acc:.4f}")

    print("Rebuilding vocabulary on full training set...")
    kmeans = build_vocab(all_paths, vocab_size=VOCAB_SIZE)

    print("Extracting BoVW+SPM features for full training set...")
    X_full = extract_features(all_paths, kmeans)

    print("Training final classifier on full training set...")
    final_clf = OneVsRestClassifier(
        LinearSVC(C=best_C, class_weight="balanced", max_iter=20000),
        n_jobs=None
    )
    final_clf.fit(X_full, all_labels_enc)

    train_acc = accuracy_score(all_labels_enc, final_clf.predict(X_full))
    print(f"Training accuracy: {train_acc:.4f}")

    print("Loading test data...")
    test_paths, filenames = load_test_dataset(test_path)

    print("Extracting BoVW+SPM features for test data...")
    X_test = extract_features(test_paths, kmeans)

    print("Predicting on test data...")
    pred_enc = final_clf.predict(X_test)
    pred_labels = le.inverse_transform(pred_enc)

    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, pred_labels):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")


if __name__ == "__main__":

    run_bovw_sift_spm("training", "testing", run_number=3)