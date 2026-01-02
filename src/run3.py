import numpy as np
from pathlib import Path

# Image loading and preprocessing
from skimage.io import imread
from skimage.color import rgb2gray

# Machine learning utilities
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# OpenCV is used for Gabor filters and HOG feature extraction
import cv2


# =====================================================================
# Paths and configuration
# =====================================================================

# Resolve project paths relative to this script
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

# Supported image file extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")


# =====================================================================
# Dataset loading
# =====================================================================

def load_training_dataset(folder):
    """
    Loads all training images and their labels.

    Expected directory structure:
        training/
            class_name/
                image.jpg

    :param folder: Path to the training dataset directory
    :return images: List of grayscale training images
    :return labels: List of class labels corresponding to each image
    """
    folder = Path(folder)
    images = []
    labels = []

    # Iterate over each class directory
    for class_dir in sorted(folder.iterdir()):
        # Skip non-directories and hidden folders
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        # Folder name is used as the class label
        label = class_dir.name

        # Load all images belonging to this class
        for img_path in sorted(class_dir.iterdir()):
            # Skip non-image files
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            # Read image from disk
            img = imread(img_path)

            # Ensure image is grayscale
            if img.ndim == 3:
                img = rgb2gray(img)

            # Store image as float for numerical stability
            images.append(img.astype(np.float32))
            labels.append(label)

    return images, labels


def load_test_dataset(folder):
    """
    Loads test images only (no labels).

    Filenames are preserved to produce output in the required format:
        <image_name> <predicted_class>

    :param folder: Path to the test dataset directory
    :return images: List of grayscale test images
    :return filenames: Array of image filenames in sorted order
    """
    folder = Path(folder)
    images = []
    filenames = []

    # Sort test images numerically to match expected submission order
    sorted_img_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS],
        key=lambda p: int(p.stem)
    )

    for img_path in sorted_img_paths:
        # Load image from disk
        img = imread(img_path)

        # Convert to grayscale if needed
        if img.ndim == 3:
            img = rgb2gray(img)

        # Store image data and corresponding filename
        images.append(img.astype(np.float32))
        filenames.append(img_path.name)

    return images, np.array(filenames)


# =====================================================================
# Feature extraction: GIST + HOG
# =====================================================================

def build_gabor_filters():
    """
    Builds a bank of Gabor filters for GIST feature extraction.

    Multiple orientations and scales are used to capture
    dominant scene layout and texture patterns.

    :return filters: List of Gabor filter kernels
    """
    filters = []

    # Kernel size chosen to emphasise global structure
    ksize = 31

    # Loop over orientations (directional sensitivity)
    for theta in np.arange(0, np.pi, np.pi / 8):

        # Loop over scales (spatial frequency sensitivity)
        for sigma in (2, 4, 8, 16, 32):
            kernel = cv2.getGaborKernel(
                (ksize, ksize),
                sigma=sigma,
                theta=theta,
                lambd=10.0,
                gamma=0.5,
                psi=0,
                ktype=cv2.CV_32F
            )
            filters.append(kernel)

    return filters


def image_hog(img):
    """
    Computes a Histogram of Oriented Gradients (HOG) descriptor.

    HOG captures local edge and shape information,
    complementing the global GIST representation.

    :param img: Input grayscale image
    :return hog_features: 1D HOG feature vector
    """
    # Ensure fixed image size required by OpenCV HOG
    if img.shape != (256, 256):
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

    # Normalise pixel intensities to [0, 255]
    img = img - img.min()
    img = img / (img.max() + 1e-6)
    img = (img * 255).astype(np.uint8)

    # Define HOG descriptor parameters
    hog = cv2.HOGDescriptor(
        _winSize=(256, 256),
        _blockSize=(64, 64),
        _blockStride=(64, 64),
        _cellSize=(16, 16),
        _nbins=9
    )

    # Compute and flatten HOG feature vector
    return hog.compute(img).flatten()


def image_gist(img, filters, n_blocks=5):
    """
    Computes a GIST descriptor for a single image.

    Gabor filter responses are spatially pooled over a grid
    to capture coarse scene layout.

    :param img: Input grayscale image
    :param filters: Gabor filter bank
    :param n_blocks: Number of spatial blocks per dimension
    :return gist_features: 1D GIST feature vector
    """
    # Zero-mean, unit-variance normalisation
    img = img - img.mean()
    img = img / (img.std() + 1e-6)

    h, w = img.shape
    block_h = h // n_blocks
    block_w = w // n_blocks

    features = []

    # Apply each Gabor filter
    for kernel in filters:
        response = cv2.filter2D(img, cv2.CV_32F, kernel)

        # Pool responses over spatial grid
        for y in range(n_blocks):
            for x in range(n_blocks):
                block = response[
                    y * block_h:(y + 1) * block_h,
                    x * block_w:(x + 1) * block_w
                ]
                # Mean absolute response for robustness to sign
                features.append(np.mean(np.abs(block)))

    return np.array(features, dtype=np.float32)


# =====================================================================
# Classifier training
# =====================================================================

def tune_and_train_svm(X, y):
    """
    Trains a linear SVM classifier.

    A validation split is used to select the regularisation
    parameter C before retraining on all training data.

    :param X: Feature matrix for training images
    :param y: Corresponding class labels
    :return final_clf: Trained linear SVM classifier
    :return best_acc: Best validation accuracy achieved
    """
    # Split training data into train/validation subsets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0,
        stratify=y
    )

    # Candidate values for SVM regularisation
    C_values = [0.01, 0.1, 1.0]
    best_C = C_values[0]
    best_acc = 0.0

    for C in C_values:
        print(f"Training Linear SVM with C={C}...")

        clf = make_pipeline(
            # Feature scaling improves optimisation stability
            StandardScaler(with_mean=False),
            # Linear SVM is efficient for high-dimensional features
            LinearSVC(C=C, max_iter=5000, class_weight="balanced")
        )

        # Train on training split
        clf.fit(X_train, y_train)

        # Evaluate on validation split
        val_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, val_pred)

        print(f"  Validation accuracy: {acc:.4f}")

        # Keep best performing model
        if acc > best_acc:
            best_acc = acc
            best_C = C

    print(f"Best C on validation: {best_C} (acc = {best_acc:.4f})")
    print("Retraining final model on all training data...")

    # Retrain classifier using the selected C on full dataset
    final_clf = make_pipeline(
        StandardScaler(with_mean=False),
        LinearSVC(C=best_C, max_iter=5000, class_weight="balanced")
    )
    final_clf.fit(X, y)

    return final_clf, best_acc


# =====================================================================
# Main experiment runner (Run #3)
# =====================================================================

def run_gist(train_dir, test_dir, run_number=3):
    """
    Complete pipeline for Run #3:
    - load datasets
    - extract GIST + HOG features
    - train linear SVM
    - predict test labels
    - save results to run3.txt

    :param train_dir: Relative path to training dataset
    :param test_dir: Relative path to test dataset
    :param run_number: Run identifier used for output filename
    """
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    train_images, train_labels = load_training_dataset(train_path)

    print("Building Gabor filter bank...")
    filters = build_gabor_filters()

    print("Extracting features for training...")
    X_train = np.array([
        # Concatenate global (GIST) and local (HOG) features
        np.concatenate([image_gist(img, filters), image_hog(img)])
        for img in train_images
    ])
    y_train = np.array(train_labels)

    print("Feature matrix shape:", X_train.shape)

    print("Training classifier...")
    clf, val_acc = tune_and_train_svm(X_train, y_train)

    # Report training accuracy to assess overfitting
    train_acc = clf.score(X_train, y_train)
    print("Training accuracy:", train_acc)
    print(f"Final validation accuracy (run {run_number}): {val_acc:.4f}")

    print("Loading test data...")
    test_images, filenames = load_test_dataset(test_path)

    print("Extracting features for test data...")
    X_test = np.array([
        np.concatenate([image_gist(img, filters), image_hog(img)])
        for img in test_images
    ])

    print("Predicting on test data...")
    predictions = clf.predict(X_test)

    # Write predictions in required submission format
    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")


if __name__ == "__main__":
    run_gist("training", "testing")
