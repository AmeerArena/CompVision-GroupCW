import numpy as np
from pathlib import Path

from skimage.io import imread
from skimage.color import rgb2gray

from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

import cv2  # OpenCV for SIFT

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
VALID_EXTS = (".jpg", ".jpeg", ".png")

# Hyperparameters for SIFT-BoVW
CLUSTERS_NUMBER = 600  # number of visual words
DENSE_STEP = 8       # step size for dense SIFT
MAX_DESC_PER_IMAGE = 500   # max descriptors to sample per image for vocab
MAX_IMAGE_SIZE = 256    # max size for the longest image side


def load_training_dataset(folder):
    """
    Loads training images and their labels.
    Assumes structure: folder/class_name/image.jpg
    Returns:
        images: list of 2D float images
        labels: list of strings
    """
    folder = Path(folder)
    images, labels = [], []

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue

        label = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            img = imread(img_path)
            
            if img.ndim == 3:
                img = rgb2gray(img)
            img = img.astype(np.float32)

            images.append(img)
            labels.append(label)

    return images, labels


def load_test_dataset(folder):
    """
    Loads test images and filenames.
    Assumes: folder/*.jpg with numeric filenames (0.jpg, 1.jpg, ...)
    Returns:
        images: list of 2D float images
        filenames: np.array of strings
    """
    folder = Path(folder)
    images, filenames = [], []

    sorted_img_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS],
        key=lambda p: int(p.stem),
    )

    for img_path in sorted_img_paths:
        img = imread(img_path)
        if img.ndim == 3:
            img = rgb2gray(img)
        img = img.astype(np.float32)

        images.append(img)
        filenames.append(img_path.name)

    return images, np.array(filenames)


def _resize_if_needed(img, max_size=MAX_IMAGE_SIZE):
    """
    Resize image so the longest side is at most max_size.
    """
    h, w = img.shape
    longest = max(h, w)
    if longest <= max_size:
        return img

    scale = max_size / float(longest)
    new_h = int(round(h * scale))
    new_w = int(round(w * scale))
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img_resized


def dense_sift(img, step=DENSE_STEP):
    """
    Compute Dense SIFT descriptors on a grid over the image.

    Args:
        img: 2D float image
        step: grid spacing in pixels

    Returns:
        descriptors: (N, 128) float32 array (can be empty)
    """
    # convert to uint8 for OpenCV
    if img.dtype != np.uint8:
        img = (img * 255.0).clip(0, 255).astype(np.uint8)

    img = _resize_if_needed(img, MAX_IMAGE_SIZE)

    h, w = img.shape
    keypoints = []
    for y in range(0, h, step):
        for x in range(0, w, step):
            keypoints.append(cv2.KeyPoint(float(x), float(y), step))

    if not keypoints:
        return np.empty((0, 128), dtype=np.float32)

    sift = cv2.SIFT_create()
    _, descriptors = sift.compute(img, keypoints)

    if descriptors is None:
        return np.empty((0, 128), dtype=np.float32)

    return descriptors.astype(np.float32)


def select_images_for_vocab(images, per_image_desc=MAX_DESC_PER_IMAGE):
    """
    For vocabulary building: extract up to per_image_desc descriptors per image.
    """
    all_desc = []
    rng = np.random.default_rng(0)

    for img in images:
        desc = dense_sift(img)
        if desc.shape[0] == 0:
            continue

        if desc.shape[0] > per_image_desc:
            idx = rng.choice(desc.shape[0], size=per_image_desc, replace=False)
            desc = desc[idx]

        all_desc.append(desc)

    if not all_desc:
        raise RuntimeError("No SIFT descriptors extracted for vocabulary!")

    return np.vstack(all_desc)


def build_vocab(train_images, clusters_number=CLUSTERS_NUMBER):
    """
    Build visual vocabulary with MiniBatchKMeans on SIFT descriptors from training images.
    """
    print("Extracting SIFT descriptors for vocabulary...")
    all_desc = select_images_for_vocab(train_images)

    print(f"Total descriptors for vocab: {all_desc.shape[0]}")
    print("Clustering descriptors with MiniBatchKMeans...")

    kmeans = MiniBatchKMeans(
        n_clusters=clusters_number,
        batch_size=1000,
        random_state=0,
        init="k-means++",
        n_init=1,
    )
    kmeans.fit(all_desc)
    return kmeans


def image_bovw_from_sift(img, kmeans):
    """
    Compute Bag-of-Visual-Words histogram from Dense SIFT using a trained KMeans vocabulary.
    """
    desc = dense_sift(img)
    if desc.shape[0] == 0:
        return np.zeros(CLUSTERS_NUMBER, dtype=np.float32)

    words = kmeans.predict(desc)
    hist, _ = np.histogram(words, bins=CLUSTERS_NUMBER, range=(0, CLUSTERS_NUMBER))

    hist = hist.astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-6)  #L2 normalisation
    return hist


def run_sift_bovw(train_dir, test_dir, run_number=3):
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    train_images, train_labels = load_training_dataset(train_path)

    print("Building SIFT vocabulary...")
    kmeans = build_vocab(train_images)

    print("Extracting SIFT-BoVW features for training...")
    X_train = np.array([image_bovw_from_sift(img, kmeans) for img in train_images])
    y_train = np.array(train_labels)

    print("Training linear SVM classifier...")
    clf = make_pipeline(
        StandardScaler(),
        LinearSVC(C=1.0, max_iter=5000),
    )
    clf.fit(X_train, y_train)

    print("Loading test data...")
    test_images, filenames = load_test_dataset(test_path)

    print("Predicting on test data...")
    predictions = []
    for img in test_images:
        feat = image_bovw_from_sift(img, kmeans).reshape(1, -1)
        pred = clf.predict(feat)[0]
        predictions.append(pred)

    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")


if __name__ == "__main__":
    run_sift_bovw("training", "testing")
