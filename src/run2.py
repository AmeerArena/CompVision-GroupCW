import numpy as np
from pathlib import Path
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
VALID_EXTS = (".jpg", ".jpeg", ".png")

CLUSTERS_NUMBER = 500 # 500 clusters as told

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

# 8x8 patches every 4 pixels
def extract_patches(img, size=8, step=4):
    """
    Gets the 8×8 patches from the img
    then flatten the patches into vectors
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

def load_training_dataset(folder):
    """
    Helper function which loads the training from a given folder, assumes training
    images are under subfolders named after their class.
    """
    folder = Path(folder)
    all_images = []  # store full paths for vocabulary building

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
            all_images.append((img, label))

    return all_images

def load_test_dataset(folder):
    """
    Helper function which loads the testing data from a given folder. folder/*.jpeg/png/jpg
    """
    folder = Path(folder)
    X, filenames = [], []

    # sort the images
    sorted_img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS], key=lambda p: int(p.stem))

    for img_path in sorted_img_paths:
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        
        img = imread(img_path)
        X.append(img.astype(np.float32))
        filenames.append(img_path.name)

    return X, np.array(filenames)

def select_images_per_class(train_data, per_class=20):
    """
    Helper function to randomly select a number of images from each category
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
    Get a sample of the images and cluster them using k-means to build the vocabulary
    """
    patch_samples = []

    imgs = select_images_per_class(train_data)

    # extract patches from each selected image
    for img in imgs:
        patches = extract_patches(img)

        # set a max of 200 patches per image
        patches = patches[:200]

        patch_samples.append(patches)

    # stack the samples into one array
    patch_samples = np.vstack(patch_samples)

    print("Clustering patches...")
    # cluster using kmeans (MiniBatchKMeans for efficincy)
    kmeans = MiniBatchKMeans(n_clusters=clusters_number, batch_size=1000, random_state=0, init="k-means++", n_init=1)
    kmeans.fit(patch_samples)
    return kmeans

def image_bovw(img, kmeans):
    
    patches = extract_patches(img)
    
    # if no patches return a zero histogram
    if len(patches) == 0:
        return np.zeros(CLUSTERS_NUMBER)

    # map each patch to nearest word (cluster center)
    words = kmeans.predict(patches)
    # make the bag-of-visual-words
    hist, _ = np.histogram(words, bins=CLUSTERS_NUMBER, range=(0, CLUSTERS_NUMBER))

    hist = hist.astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-6)  # normalise
    return hist


def train_binary_classifer(X_train, y_train, target_class):

    # Label the correct class as 1, label the rest as 0
    y_binary = (y_train == target_class).astype(int)

    # Train a linear SVM
    clf = LinearSVC()
    clf.fit(X_train, y_binary)

    return clf


def predict_one_vs_all(classifiers, X_input):
    scores = {}
    # Get the decision score from each binary classifier
    for class_label, clf in classifiers.items():
        score = clf.decision_function(X_input)[0]
        scores[class_label] = score

    # Pick class with highest decision score
    predicted_class = max(scores, key=scores.get)
    return predicted_class


def run_bovw(train_dir, test_dir, run_number=2):
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    train_data = load_training_dataset(train_path)
    images, labels = zip(*train_data)

    print("Building vocabulary...")
    kmeans = build_vocab(train_data)

    print("Extracting training BoVW features...")
    X_train = np.array([image_bovw(img, kmeans) for img in images]) # matrix
    y_train = np.array(labels)


    # This is not what we are supposed to be training one classifer, we need to train 15 one vs classifier
    # print("Training linear classifiers (one-vs-all)...")
    # clf = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", max_iter=4000, tol=1e-3, n_jobs=-1))
    # clf.fit(X_train, y_train)

    classes = sorted(set(y_train))
    classifiers = {}
        
    for c in classes:
        print("Training classifier for class: {c}")
        classifier = train_binary_classifer(X_train, y_train, c)
        classifiers[c] = classifier


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

import numpy as np
from pathlib import Path
from skimage.io import imread
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

FILE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = FILE_DIR.parent
VALID_EXTS = (".jpg", ".jpeg", ".png")

CLUSTERS_NUMBER = 500 # 500 clusters as told

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

# 8x8 patches every 4 pixels
def extract_patches(img, size=8, step=4):
    """
    Gets the 8×8 patches from the img
    then flatten the patches into vectors
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

def load_training_dataset(folder):
    """
    Helper function which loads the training from a given folder, assumes training
    images are under subfolders named after their class.
    """
    folder = Path(folder)
    all_images = []  # store full paths for vocabulary building

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
            all_images.append((img, label))

    return all_images

def load_test_dataset(folder):
    """
    Helper function which loads the testing data from a given folder. folder/*.jpeg/png/jpg
    """
    folder = Path(folder)
    X, filenames = [], []

    # sort the images
    sorted_img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in VALID_EXTS], key=lambda p: int(p.stem))

    for img_path in sorted_img_paths:
        if img_path.suffix.lower() not in VALID_EXTS:
            continue
        
        img = imread(img_path)
        X.append(img.astype(np.float32))
        filenames.append(img_path.name)

    return X, np.array(filenames)

def select_images_per_class(train_data, per_class=60):
    """
    Helper function to randomly select a number of images from each category
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
    Get a sample of the images and cluster them using k-means to build the vocabulary
    """
    patch_samples = []

    imgs = select_images_per_class(train_data)

    # extract patches from each selected image
    for img in imgs:
        patches = extract_patches(img)

        # set a max of 200 patches per image
        patches = patches[:400]

        patch_samples.append(patches)

    # stack the samples into one array
    patch_samples = np.vstack(patch_samples)

    print("Clustering patches...")
    # cluster using kmeans (MiniBatchKMeans for efficincy)
    kmeans = MiniBatchKMeans(n_clusters=clusters_number, batch_size=1000, random_state=0, init="k-means++", n_init=1)
    kmeans.fit(patch_samples)
    return kmeans

def image_bovw(img, kmeans):
    
    patches = extract_patches(img)
    
    # if no patches return a zero histogram
    if len(patches) == 0:
        return np.zeros(CLUSTERS_NUMBER)

    # map each patch to nearest word (cluster center)
    words = kmeans.predict(patches)
    # make the bag-of-visual-words
    hist, _ = np.histogram(words, bins=CLUSTERS_NUMBER, range=(0, CLUSTERS_NUMBER))

    hist = hist.astype(np.float32)
    hist /= (np.linalg.norm(hist) + 1e-6)  # normalise
    return hist


def train_binary_classifer(X_train, y_train, target_class):

    # Label the correct class as 1, label the rest as 0
    y_binary = (y_train == target_class).astype(int)

    # Train a linear SVM
    clf = LinearSVC()
    clf.fit(X_train, y_binary)

    return clf


def test_binary_classifiers(classifiers, X_input):
    scores = {}
    # Get the decision score from each binary classifier
    for class_label, clf in classifiers.items():
        score = clf.decision_function(X_input)[0]
        scores[class_label] = score

    # Pick class with highest decision score
    predicted_class = max(scores, key=scores.get)
    return predicted_class


def run_bovw(train_dir, test_dir, run_number=2):
    train_path = PROJECT_ROOT / train_dir
    test_path = PROJECT_ROOT / test_dir

    print("Loading training data...")
    train_data = load_training_dataset(train_path)
    images, labels = zip(*train_data)

    print("Building vocabulary...")
    kmeans = build_vocab(train_data)

    print("Extracting training BoVW features...")
    X_train = np.array([image_bovw(img, kmeans) for img in images]) # matrix
    y_train = np.array(labels)


    # This is not what we are supposed to be training one classifer, we need to train 15 one vs classifier
    # print("Training linear classifiers (one-vs-all)...")
    # clf = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", max_iter=4000, tol=1e-3, n_jobs=-1))
    # clf.fit(X_train, y_train)

    # Classes e.g. [bedroom, Coast, Forest . . .]
    classes = sorted(set(y_train))

    # Each class will have it's own specific classifier (One vs All)
    classifiers = {}
        
    for c in classes:
        print("Training classifier for class:", c)
        classifier = train_binary_classifer(X_train, y_train, c)
        classifiers[c] = classifier


    print("Loading test data...")
    test_imgs, filenames = load_test_dataset(test_path)

    print("Predicting...")
    predictions = []

    for  index,img in enumerate(test_imgs):
        print("Predicting class for image:", index)
        feat = image_bovw(img, kmeans).reshape(1, -1)
        predicted = test_binary_classifiers(classifiers, feat)
        print("Predicted class for image:", index, "is:", predicted)
        predictions.append(predicted)

    # Save predictions
    output_name = f"run{run_number}.txt"
    with open(output_name, "w") as f:
        for fname, pred in zip(filenames, predictions):
            f.write(f"{fname} {pred}\n")

    print(f"Saved predictions to {output_name}")

if __name__ == "__main__":
    run_bovw("training", "testing", run_number=2)
    

