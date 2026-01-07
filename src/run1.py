import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

# Resolve project paths relative to this script
FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

# Supported image file extensions
VALID_EXTS = (".jpg", ".jpeg", ".png")


def tiny_image(img, size=16) -> np.ndarray:
    """
    We turn an image into a "tiny image" feature vector.

    we do this by the following steps:
      1) centre-crop the image to a square (so width==height)
      2) resize that square down to size x size (default 16x16)
      3) flatten pixels into one long vector
      4) normalise the vector (zero mean and unit length)

    Args:
        img: grayscale image as a 2D numpy array (H x W)
        size: tiny image side length (eg, 16 means 16x16)
        
    Returns:
        np.ndarray: The processed tiny image as a flattened and normalized vector (1D numpy array of length size*size (eg 256 values for 16x16))
    """

    # img.shape[:2] gives (height, width). Works even if img has extra channels,
    # but we load grayscale so it should be 2D.
    h, w = img.shape[:2]
    
    #we crop a centred square using the smaller side length
    side = min(h, w)

   #top-left corner of the centre square
    y0 = (h - side) // 2
    x0 = (w - side) // 2

    #crop: we keep only the central square region
    img = img[y0:y0+side, x0:x0+side]

    #resizes the square to (size, size)
    # anti_aliasing=True helps reduce jagged artifacts when shrinking images
    img = resize(img, (size, size), anti_aliasing=True).astype(np.float32)
    
     #flatten from 2D (size x size) into 1D (size*size)
    vec = img.flatten()

    # normalisation step 1: subtract the mean (zero-mean)
    vec -= vec.mean()

    #normalisation step 2: divide by its length (unit-length)
    #this avoids larger-contrast images dominating the distance metric
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n

    return vec

# =====================================================================
# Dataset loading
# =====================================================================

def load_training_dataset(folder, size=16) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function which loads the training from a given folder, assumes training
    images are under subfolders named after their class.

    Args:
        folder (str): Path to the training dataset folder
        size (int, optional): Size to which images are resized. Defaults to 16.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature vectors and labels array.
    """
    folder = Path(folder)

    # we'll build lists first, then convert to numpy arrays at the end
    X, y = [], []

    # sorted(...) makes the order deterministic (same every run)
    for class_dir in sorted(folder.iterdir()):
        #skip files, only use directories where each directory = one class
        if not class_dir.is_dir():
            continue

        #skip hidden directories (those starting with .)
        if class_dir.name.startswith("."):
            continue

        #folder name is the class label (eg: "forest", "coast", etc)
        label = class_dir.name

        #iterates images inside the class folder
        for img_path in sorted(class_dir.iterdir()):

            # ignores non-image files
            if img_path.suffix.lower() not in VALID_EXTS:
                continue

            #read as grayscale (2D float-ish image)
            img = imread(img_path, as_gray=True)

            # Convert to tiny image feature
            X.append(tiny_image(img, size))
            y.append(label)

    return np.array(X), np.array(y)


def load_test_dataset(folder, size=16) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function which loads the testing data from a given folder. folder/*.jpeg/png/jpg

     We must output predictions in the format:
        <image_name> <predicted_class>

     We sort numerically by filename stem to keep a clean order.

    Args:
        folder (str): Path to the testing dataset folder
        size (int, optional): Size to which images are resized. Defaults to 16.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature vectors and filenames array.
    """
    folder = Path(folder)
    X, filenames = [], []
    
    #Collect valid images, then sort by numeric filename (eg: 0,1,2,...)
    # p.stem is filename without extension so "123.jpg"->"123"
    sorted_img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS], key=lambda p: int(p.stem))

    for img_path in sorted_img_paths:
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        img = imread(img_path, as_gray=True)
        X.append(tiny_image(img, size))
        filenames.append(img_path.name)

    return np.array(X), np.array(filenames)

# =====================================================================
# Main experiment runner (Run #1)
# =====================================================================

def run_knn(train_dir, test_dir, run_number=1, k=3, size=16):
    """
    Run k nearest neighbours on the training data and testing data

    Full pipeline for Run #1:
      - load training images and compute tiny-image features
      - load test images and compute tiny-image features
      - train a KNN classifier on the training set
      - predict labels for the test set
      - write predictions into run1.txt (or run{run_number}.txt)

    Args:
        train_dir (str): Path to the training dataset folder
        test_dir (str): Path to the testing dataset folder
        run_number (int, optional): Run number identifier. Defaults to 1.
        k (int, optional): Number of neighbors for KNN. Defaults to 3.
        size (int, optional): Size to which images are resized. Defaults to 16.
    """
    #builds absolute paths from project root
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    X, y = load_training_dataset(train_path, size)

    # 80/20 train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y  # keeps class balance
    )

    print("Loading test data...")
    X_test, filenames = load_test_dataset(test_path, size)

    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    print("Evaluating on validation set...")
    val_preds = knn.predict(X_val)
    accuracy = accuracy_score(y_val, val_preds)

    print(f"Validation accuracy: {accuracy:.4f}")

    print("Predicting...")
    predictions = knn.predict(X_test)

    # Save predictions
    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")


if __name__ == "__main__":
    run_knn("training", "testing", k=5, size=16)
