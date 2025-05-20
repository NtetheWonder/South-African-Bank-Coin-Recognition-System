import os
import cv2
import numpy as np
import logging
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

# ------------------------------
# Configuration
# ------------------------------
CLASS_LABELS = ['5c', '10c', '20c', '50c', 'R1', 'R2', 'R5']
DATASET_PATH = "dataset"
TEST_IMAGE_PATH = "test_images/10c_O.png"
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
DEBUG = False

# ------------------------------
# Logging Setup
# ------------------------------
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Convert to grayscale, normalize lighting, and remove noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    norm = cv2.equalizeHist(gray)
    blur = cv2.medianBlur(norm, 5)
    return blur

# ------------------------------
# Segmentation
# ------------------------------
def segment_coin(image: np.ndarray) -> np.ndarray:
    """
    Detect circular coin region using Hough Transform and return mask.
    """
    circles = cv2.HoughCircles(
        image,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=100,
        param2=30,
        minRadius=30,
        maxRadius=150
    )
    mask = np.zeros_like(image)
    if circles is not None:
        x, y, r = np.uint16(circles[0][0])
        cv2.circle(mask, (x, y), r, 255, -1)
    return mask if mask.sum() > 0 else None

# ------------------------------
# Feature Extraction
# ------------------------------
def extract_shape_features(mask: np.ndarray) -> np.ndarray:
    """
    Compute aspect ratio and extent from the largest contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.array([0.0, 0.0])
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect_ratio = float(w) / h
    extent = cv2.contourArea(cnt) / (w * h)
    return np.array([aspect_ratio, extent])


def extract_features(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Extract Hu Moments, color histogram, LBP, and shape features.
    """
    # Apply mask
    masked = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

    # Hu Moments
    moments = cv2.moments(mask)
    hu = cv2.HuMoments(moments).flatten()

    # Color histogram
    hist = cv2.calcHist([masked], [0, 1, 2], mask, [8, 8, 8], [0, 256]*3)
    hist = cv2.normalize(hist, hist).flatten()

    # LBP histogram
    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    (hist_lbp, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_N_POINTS + 3),
        range=(0, LBP_N_POINTS + 2)
    )
    hist_lbp = hist_lbp.astype('float')
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # Shape features
    shape_feats = extract_shape_features(mask)

    # Debug display
    if DEBUG:
        cv2.imshow("Masked", masked)
        cv2.waitKey(0)

    return np.hstack([hu, hist, hist_lbp, shape_feats])

# ------------------------------
# Dataset Loading
# ------------------------------
def load_dataset(path: str):
    X, y = [], []
    for label in os.listdir(path):
        lab_dir = os.path.join(path, label)
        if not os.path.isdir(lab_dir):
            continue
        for fname in os.listdir(lab_dir):
            img = cv2.imread(os.path.join(lab_dir, fname))
            if img is None:
                continue
            proc = preprocess_image(img)
            mask = segment_coin(proc)
            if mask is None:
                continue
            feats = extract_features(img, mask)
            X.append(feats)
            y.append(label)
    return np.array(X), np.array(y)

# ------------------------------
# Training & Evaluation
# ------------------------------
def train_and_evaluate(X, y):
    # Stratified train-test split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for tr_idx, te_idx in split.split(X, y):
        X_train, X_test = X[tr_idx], X[te_idx]
        y_train, y_test = y[tr_idx], y[te_idx]

    # Model pipeline
    model = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=42))
    model.fit(X_train, y_train)

    # Report
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, labels=CLASS_LABELS, target_names=CLASS_LABELS, zero_division=0)
    logging.info("Classification Report:\n%s", report)
    return model

# ------------------------------
# Prediction on New Image
# ------------------------------
def predict_image(model, image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        logging.error("Failed to load image: %s", image_path)
        return
    proc = preprocess_image(img)
    mask = segment_coin(proc)
    if mask is None:
        logging.warning("No coin detected in %s", image_path)
        return
    feat = extract_features(img, mask).reshape(1, -1)
    pred = model.predict(feat)[0]
    prob = model.predict_proba(feat).max()
    logging.info("Predicted: %s (%.2f%%)", pred, prob*100)
    cv2.putText(img, f"{pred} ({prob*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Prediction", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# Main
# ------------------------------
def main():
    logging.info("Loading dataset from %s...", DATASET_PATH)
    X, y = load_dataset(DATASET_PATH)
    logging.info("Loaded %d samples.", len(X))
    model = train_and_evaluate(X, y)
    logging.info("Predicting on new image: %s", TEST_IMAGE_PATH)
    predict_image(model, TEST_IMAGE_PATH)

if __name__ == '__main__':
    main()
