import os
import cv2
import numpy as np
import streamlit as st
import logging
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from PIL import Image

# ------------------------------
# Configuration
# ------------------------------
CLASS_LABELS = ['5c', '10c', '20c', '50c', 'R1', 'R2', 'R5']
DATASET_PATH = 'dataset'
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
RESIZE_SHAPE = (128, 128)

# ------------------------------
# Logging
# ------------------------------
logging.basicConfig(level=logging.INFO)

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_image(image):
    image = cv2.resize(image, RESIZE_SHAPE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5,5), 0)
    return blur

# ------------------------------
# Segmentation
# ------------------------------
def segment_coin(image):
    _, thr = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    cnt = max(contours, key=cv2.contourArea)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    return mask

# ------------------------------
# Feature Extraction
# ------------------------------
def extract_shape_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0, 0.0, 0.0]

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = w / h
    extent = cv2.contourArea(cnt) / (w * h)
    area = cv2.contourArea(cnt)
    return [aspect, extent, area]

def extract_features(preproc, mask):
    masked = cv2.bitwise_and(preproc, preproc, mask=mask)

    lbp = local_binary_pattern(masked, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    hist, _ = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, LBP_N_POINTS + 3),
        range=(0, LBP_N_POINTS + 2)
    )
    hist = hist.astype(float)
    hist /= (hist.sum() + 1e-7)

    hog_feats = hog(
        masked,
        orientations=9,
        pixels_per_cell=(16,16),
        cells_per_block=(2,2),
        block_norm='L2-Hys'
    )

    shape_feats = extract_shape_features(mask)
    return np.hstack([hist, hog_feats, shape_feats])

# ------------------------------
# Dataset Loader
# ------------------------------
def load_dataset(path):
    X, y = [], []
    for label in CLASS_LABELS:
        folder = os.path.join(path, label)
        if not os.path.isdir(folder):
            continue

        for file in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, file))
            if img is None:
                continue

            proc = preprocess_image(img)
            mask = segment_coin(proc)
            if mask is None:
                continue

            feats = extract_features(proc, mask)
            X.append(feats)
            y.append(label)

    return np.array(X), np.array(y)

# ------------------------------
# Train Model (Cached)
# ------------------------------
@st.cache_resource
def train_model():
    X, y = load_dataset(DATASET_PATH)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(sss.split(X, y))

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X[train_idx], y[train_idx])
    return pipeline

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Coin Classifier", layout="centered")
st.title("🪙 Coin Classification System")
st.write("Upload a coin image to predict its denomination.")

model = train_model()

uploaded_file = st.file_uploader(
    "Upload a coin image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    proc = preprocess_image(img_np)
    mask = segment_coin(proc)

    if mask is None:
        st.error("❌ No coin detected in the image.")
    else:
        feats = extract_features(proc, mask).reshape(1, -1)
        prediction = model.predict(feats)[0]
        confidence = model.predict_proba(feats).max()

        st.success(f"### 🏷️ Prediction: **{prediction}**")
        st.info(f"Confidence: **{confidence * 100:.2f}%**")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Preprocessed Image")
            st.image(proc, clamp=True)

        with col2:
            st.subheader("Segmentation Mask")
            st.image(mask, clamp=True)

st.markdown("---")
st.caption("Built with Streamlit · OpenCV · scikit-learn")

