import numpy as np
from pathlib import Path
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# =====================================================================
# Paths and configuration
# =====================================================================

# Resolve project paths relative to this script
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

# Supported image file extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")

# 500 visual words (k-means clusters)
CLUSTERS_NUMBER = 500 

# Debugging function
def bedroom_image(index):
    """
    Loads the index-th image in a given folder.
    Returns the image as a NumPy array.
    """

    path =  "/Users/user/Desktop/scene_recognition/training/bedroom"
    folder = Path(path)

    img_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS]
    )

    if index < 0 or index >= len(img_paths):
        raise IndexError(f"Index {index} out of range for folder {path}")

    
    img = imread(img_paths[index], as_gray=True).astype(np.float32)

    return img

# =====================================================================
# Patch extraction
# =====================================================================

def extract_patches(img, size=8, step=4):
    """
    Extracts overlapping 8×8 grayscale patches from an image.

    Patches are extracted every 4 pixels then flattened into vectors

    :param img: Input grayscale image
    :param size: Patch size (default: 8×8)
    :param step: Pixel distance between patches (default: 4 pixels)
    :return patches: Array of flattened image patches
    """
    img = img.astype(np.float32)
    h, w = img.shape
    patches = []
    for y in range(0, h - size + 1, step):
        for x in range(0, w - size + 1, step):
            # flatten the patches
            patch = img[y:y+size, x:x+size].reshape(-1)
            patch -= patch.mean()  
            patches.append(patch)

    return np.array(patches, dtype=np.float32)

# =====================================================================
# Dataset loading
# =====================================================================

def load_training_dataset(folder):
    """
    Loads the training dataset.

    Expected directory structure:
        training/
            class_name/
                image.jpg

    :param folder: Path to the training dataset directory
    :return all_images: List of (image, label) tuples
    """
    folder = Path(folder)
    all_images = []  # store full paths for vocabulary building

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

            # Load in grayscale
            img = imread(img_path, as_gray=True)
            all_images.append((img, label))

    return all_images


def load_test_dataset(folder):
    """
    Loads test images only (no labels).

    Filenames are preserved to produce output in the format:
        <image_name> <predicted_class>

    :param folder: Path to the test dataset directory
    :return images: List of test images
    :return filenames: Array of image filenames
    """
    folder = Path(folder)
    images = []
    filenames = []

    # Sort test images numerically to match expected submission order
    sorted_img_paths = sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS],
        key=lambda p: int(p.stem))

    for img_path in sorted_img_paths:
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        
        # Load image from disk
        img = imread(img_path)
        # Store image data and corresponding filename
        images.append(img.astype(np.float32))
        filenames.append(img_path.name)

    return images, np.array(filenames)


# =====================================================================
# Vocabulary construction (Bag of Visual Words)
# =====================================================================

def select_images_per_class(train_data, per_class=20):
    """
    Randomly selects a fixed number of images per class.

    :param train_data: List of (image, label) tuples
    :param per_class: Number of images per class to select
    :return selected: List of selected images
    """
    
    # Group images by class
    class_groups = {}
    for img, label in train_data:
        class_groups.setdefault(label, []).append(img)

    selected = []

    for label in sorted(class_groups.keys()):
        imgs = class_groups[label]

        # If class has fewer than per_class, take all
        if len(imgs) <= per_class:
            chosen = imgs
        else:
            # Randomly choose
            idx = np.random.choice(len(imgs), size=per_class, replace=False)
            chosen = [imgs[i] for i in idx]

        selected.extend(chosen)

    return selected


def build_vocab(train_data, clusters_number=CLUSTERS_NUMBER):
    """
    Builds the visual vocabulary using k-means clustering.

    Image patches are extracted from a subset of training images,
    then MiniBatchKMeans is used to cluster them into visual words.

    :param train_data: List of (image, label) tuples
    :param clusters_number: Number of visual words
    :return kmeans: Trained k-means model
    """
    
    patch_samples = []
    
    # Select a subset of images per class
    imgs = select_images_per_class(train_data)

    # extract patches from each selected image
    for img in imgs:
        patches = extract_patches(img)

        # set a max of 200 patches per image
        patches = patches[:200]
        patch_samples.append(patches)

    # Stack all patch vectors
    patch_samples = np.vstack(patch_samples)

    print("Clustering patches...")
    # cluster using kmeans (MiniBatchKMeans for efficincy)
    kmeans = MiniBatchKMeans(
        n_clusters=clusters_number,
        batch_size=1000,
        random_state=0,
        init="k-means++",
        n_init=1
    )
    kmeans.fit(patch_samples)
    return kmeans


def image_bovw(img, kmeans):
    """
    Computes a Bag of Visual Words (BoVW) representation for an image.

    Each image patch is assigned to the nearest visual word,
    and a normalised histogram of visual word frequencies is returned.

    :param img: Input grayscale image
    :param kmeans: Trained k-means vocabulary
    :return hist: Normalised BoVW feature vector
    """
    
    patches = extract_patches(img)
    
    if len(patches) == 0:
        return np.zeros(CLUSTERS_NUMBER)

    # Map each patch to nearest word (cluster center)
    words = kmeans.predict(patches)
    
    # Build histogram of visual words
    hist, _ = np.histogram(words, bins=CLUSTERS_NUMBER, range=(0, CLUSTERS_NUMBER))

    hist = hist.astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-6)  # normalise
    return hist


# =====================================================================
# One-vs-All classifier training
# =====================================================================

def train_binary_classifer(X_train, y_train, target_class):
    """
    Trains a binary linear SVM classifier for one-vs-all classification.
    
    :param X_train: Training feature matrix
    :param y_train: Training labels
    :param target_class: The target class
    :return clf: Trained binary classifier
    """

    # Label the correct class as 1, label the rest as 0
    y_binary = (y_train == target_class).astype(int)

    # Train a linear SVM
    clf = LinearSVC()
    clf.fit(X_train, y_binary)

    return clf


def predict_one_vs_all(classifiers, X_input):
    """
    Predicts a class label using a set of one-vs-all classifiers.

    Class with the highest decision score is selected.

    :param classifiers: Dictionary of trained classifiers
    :param X_input: Feature vector for a single image
    :return predicted_class: Predicted class label
    """
    
    scores = {}
    # Get the decision score from each binary classifier
    for class_label, clf in classifiers.items():
        score = clf.decision_function(X_input)[0]
        scores[class_label] = score

    # Pick class with highest decision score
    predicted_class = max(scores, key=scores.get)
    return predicted_class


# =====================================================================
# Main experiment runner (Run #2)
# =====================================================================

def run_bovw(train_dir, test_dir, run_number=2):
    """
    Complete pipeline for Run #2:
    - load training data
    - build visual vocabulary
    - extract BoVW features
    - train one-vs-all linear SVMs
    - predict test labels
    - save results to run2.txt

    :param train_dir: Relative path to training dataset
    :param test_dir: Relative path to test dataset
    :param run_number: Run identifier used for output filename
    """
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    train_data = load_training_dataset(train_path)
    images, labels = zip(*train_data)

    train_data = load_training_dataset(train_path)

    train_split, val_split = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in train_data]
    )

    train_images, train_labels = zip(*train_split)
    val_images, val_labels = zip(*val_split)

    print("Building vocabulary...")
    kmeans = build_vocab(train_split)

    print("Extracting training BoVW features...")
    X_train = np.array([image_bovw(img, kmeans) for img in train_images])
    y_train = np.array(train_labels)

    print("Extracting validation BoVW features...")
    X_val = np.array([image_bovw(img, kmeans) for img in val_images])
    y_val = np.array(val_labels)

    # Train one-vs-all classifiers
    classes = sorted(set(y_train))
    classifiers = {}
        
    for c in classes:
        print("Training classifier for class: {c}")
        classifier = train_binary_classifer(X_train, y_train, c)
        classifiers[c] = classifier
    
    print("Evaluating on validation set...")
    val_predictions = []

    for feat in X_val:
        pred = predict_one_vs_all(classifiers, feat.reshape(1, -1))
        val_predictions.append(pred)

    accuracy = accuracy_score(y_val, val_predictions)
    print(f"Validation accuracy: {accuracy:.4f}")

    print("Rebuilding vocabulary on full training set...")
    kmeans = build_vocab(train_data)

    X_full = np.array([image_bovw(img, kmeans) for img, _ in train_data])
    y_full = np.array([label for _, label in train_data])

    classifiers = {}
    for c in sorted(set(y_full)):
        classifiers[c] = train_binary_classifer(X_full, y_full, c)


    print("Loading test data...")
    test_imgs, filenames = load_test_dataset(test_path)

    print("Predicting...")
    predictions = []

    for img in test_imgs:
        feat = image_bovw(img, kmeans).reshape(1, -1)
        predicted = predict_one_vs_all(classifiers, feat)
        predictions.append(predicted)

    # Save predictions
    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")

def main():
    run_bovw("training", "testing", run_number=2)

if __name__ == "__main__":
    main()
