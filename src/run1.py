import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from skimage.io import imread
from skimage.transform import resize
from pathlib import Path

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent

VALID_EXTS = (".jpg", ".jpeg", ".png")


def tiny_image(img, size=16) -> np.ndarray:
    """
    Creates the tiny image using a crop of the image at centre.
    Normalises the vector (zero mean and unit length)

    Args:
        img: the image file
        size (int, optional): Dimensions to resize to Defaults to 16.

    Returns:
        np.ndarray: The processed tiny image as a flattened and normalized vector.
    """
    h, w = img.shape[:2]
    side = min(h, w)

    # centre crop
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    img = img[y0:y0+side, x0:x0+side]

    # resize
    img = resize(img, (size, size), anti_aliasing=True).astype(np.float32)
    vec = img.flatten()

    # normalise
    vec -= vec.mean()
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n

    return vec


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
    X, y = [], []

    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        if class_dir.name.startswith("."):
            continue

        label = class_dir.name

        for img_path in sorted(class_dir.iterdir()):
            if img_path.suffix.lower() not in VALID_EXTS:
                continue
            img = imread(img_path, as_gray=True)
            X.append(tiny_image(img, size))
            y.append(label)

    return np.array(X), np.array(y)


def load_test_dataset(folder, size=16) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper function which loads the testing data from a given folder. folder/*.jpeg/png/jpg

    Args:
        folder (str): Path to the testing dataset folder
        size (int, optional): Size to which images are resized. Defaults to 16.
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the feature vectors and filenames array.
    """
    folder = Path(folder)
    X, filenames = [], []
    
    # sort the images
    sorted_img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS], key=lambda p: int(p.stem))

    for img_path in sorted_img_paths:
        if img_path.suffix.lower() not in VALID_EXTS:
            continue

        img = imread(img_path, as_gray=True)
        X.append(tiny_image(img, size))
        filenames.append(img_path.name)

    return np.array(X), np.array(filenames)


def run_knn(train_dir, test_dir, run_number=1, k=3, size=16):
    """
    Run k nearest neighbours on the training data and testing data

    Args:
        train_dir (str): Path to the training dataset folder
        test_dir (str): Path to the testing dataset folder
        run_number (int, optional): Run number identifier. Defaults to 1.
        k (int, optional): Number of neighbors for KNN. Defaults to 3.
        size (int, optional): Size to which images are resized. Defaults to 16.
    """
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    X_train, y_train = load_training_dataset(train_path, size)

    print("Loading test data...")
    X_test, filenames = load_test_dataset(test_path, size)

    print("Training KNN...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

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