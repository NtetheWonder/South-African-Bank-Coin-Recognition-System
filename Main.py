import os
import cv2
import numpy as np
import logging
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from tkinter import Tk, filedialog, Button, Label, Frame, OptionMenu, StringVar
import tkinter as tk
from PIL import Image, ImageTk

# ------------------------------
# Configuration
# ------------------------------
CLASS_LABELS = ['5c', '10c', '20c', '50c', 'R1', 'R2', 'R5']
DATASET_PATH = 'dataset'
TEST_IMAGES_FOLDER = 'test_images'
LBP_RADIUS = 3
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'
RESIZE_SHAPE = (128, 128)
DEBUG = False
SAVE_OUTPUTS = True


# Logging Setup
def setup_logging():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG if DEBUG else logging.INFO
    )

# ------------------------------
# Preprocessing
# ------------------------------
def preprocess_image(image: np.ndarray) -> np.ndarray:
    image = cv2.resize(image, RESIZE_SHAPE)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    eq = clahe.apply(gray)
    blur = cv2.GaussianBlur(eq, (5,5), 0)
    return blur

# ------------------------------
# Segmentation
# ------------------------------
def segment_coin(image: np.ndarray) -> np.ndarray:
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
def extract_shape_features(mask: np.ndarray) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return [0.0, 0.0, 0.0]
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    aspect = float(w) / float(h)
    extent = cv2.contourArea(cnt) / (w * h)
    area = cv2.contourArea(cnt)
    return [aspect, extent, area]

def extract_features(preproc: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.shape != preproc.shape:
        mask = cv2.resize(mask, preproc.shape[::-1], interpolation=cv2.INTER_NEAREST)
    masked = cv2.bitwise_and(preproc, preproc, mask=mask)
    lbp = local_binary_pattern(masked, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_N_POINTS+3), range=(0, LBP_N_POINTS+2))
    hist_lbp = hist_lbp.astype(float) / (hist_lbp.sum() + 1e-7)
    hog_feats = hog(masked, orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), block_norm='L2-Hys')
    shape_feats = extract_shape_features(mask)
    return np.hstack([hist_lbp, hog_feats, shape_feats])

# ------------------------------
# Dataset Loading
# ------------------------------
def load_dataset(path: str):
    X, y = [], []
    for label in CLASS_LABELS:
        lab_dir = os.path.join(path, label)
        if not os.path.isdir(lab_dir): continue
        for fname in os.listdir(lab_dir):
            img = cv2.imread(os.path.join(lab_dir, fname))
            if img is None: continue
            proc = preprocess_image(img)
            mask = segment_coin(proc)
            if mask is None: continue
            feats = extract_features(proc, mask)
            if feats.size == 0: continue
            X.append(feats); y.append(label)
    return np.array(X), np.array(y)

# ------------------------------
# Plotting Utilities
# ------------------------------
def plot_confusion_matrix(cm, labels):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    ax.set_ylabel('True label'); ax.set_xlabel('Predicted label')
    plt.title('Confusion Matrix'); plt.tight_layout(); plt.show()

def plot_precision_recall(y_true, y_pred, labels):
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    ax.bar(x - 0.2, precision, width=0.4, label='Precision')
    ax.bar(x + 0.2, recall, width=0.4, label='Recall')
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45)
    ax.set_ylabel('Score'); ax.set_title('Per-Class Precision and Recall'); ax.legend()
    plt.tight_layout(); plt.show()

def plot_learning_curve(estimator, X, y):
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)
    train_sizes, train_scores, val_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=-1,
                                                            train_sizes=np.linspace(0.1,1.0,5), scoring='accuracy')
    train_mean = train_scores.mean(axis=1); val_mean = val_scores.mean(axis=1)
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_mean, label='Training score'); ax.plot(train_sizes, val_mean, label='Validation score')
    ax.set_xlabel('Training Examples'); ax.set_ylabel('Accuracy'); ax.set_title('Learning Curve'); ax.legend()
    plt.tight_layout(); plt.show()

def plot_roc_curves(ovr_clf, X_test, y_test, labels):
    y_bin = label_binarize(y_test, classes=labels)
    y_score = ovr_clf.predict_proba(X_test)
    fig, ax = plt.subplots()
    for i, label in enumerate(labels):
        positives = np.sum(y_bin[:, i])
        if positives < 1 or positives == len(y_bin): continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i]); roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{label} (AUC = {roc_auc:.2f})')
    ax.plot([0,1],[0,1],'--', linewidth=1)
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves (One-vs-Rest)'); ax.legend(loc='lower right')
    plt.tight_layout(); plt.show()

# ------------------------------
# Predict from Webcam
# ------------------------------
def predict_from_webcam(model):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Could not open webcam.")
        return

    logging.info("Press 'c' to capture a frame, 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab frame.")
            break
        disp = frame.copy()
        cv2.putText(disp, "Press 'c' to capture, 'q' to quit", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow("Webcam", disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            img = frame.copy()
            proc = preprocess_image(img)
            mask = segment_coin(proc)
            if mask is not None:
                feats = extract_features(proc, mask).reshape(1, -1)
                pred = model.predict(feats)[0]
                prob = model.predict_proba(feats).max()
                cv2.putText(img, f"{pred} ({prob*100:.1f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                cv2.imshow("Prediction", img)
                if SAVE_OUTPUTS:
                    cv2.imwrite(f"prediction_{pred}.jpg", img)
            else:
                logging.warning("No coin detected in captured frame.")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------
# Predict Multiple Files
# ------------------------------
def predict_multiple_images(model):
    Tk().withdraw()
    image_paths = filedialog.askopenfilenames(title="Select multiple coin images", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    for path in image_paths:
        if not path:
            continue
        img = cv2.imread(path)
        if img is None:
            logging.error(f"Cannot load image: {path}")
            continue
        original = img.copy()
        proc = preprocess_image(img)
        mask = segment_coin(proc)
        if mask is None:
            logging.warning(f"No coin detected in {path}")
            continue
        feats = extract_features(proc, mask).reshape(1, -1)
        pred = model.predict(feats)[0]; prob = model.predict_proba(feats).max()
        cv2.putText(original, f"{pred} ({prob*100:.1f}%)", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow(f"Prediction: {os.path.basename(path)}", original)
        if SAVE_OUTPUTS:
            cv2.imwrite(f"predicted_{os.path.basename(path)}", original)
        cv2.waitKey(0)
    cv2.destroyAllWindows()

# ------------------------------
# GUI Interface
# ------------------------------

def launch_interface(model):
    root = tk.Tk()
    root.title('Coin Classifier')

    frame = Frame(root)
    frame.pack(pady=10)

    result_label = Label(frame, text='Prediction will appear here', font=('Helvetica', 14, 'bold'), fg='blue')
    result_label.pack(pady=5)

    image_label = Label(frame)
    image_label.pack()

    def predict_from_file(file_path):
        img = cv2.imread(file_path)
        if img is None:
            logging.error(f'Could not read image: {file_path}')
            return
        # Stage visuals
        proc = preprocess_image(img)
        mask = segment_coin(proc)
        feats = extract_features(proc, mask).reshape(1, -1) if mask is not None else None

        if mask is None:
            result_label.config(text='No coin detected.')
        else:
            pred = model.predict(feats)[0]
            prob = model.predict_proba(feats).max()
            result_label.config(text=f'Prediction: {pred} ({prob*100:.1f}%)')

        # Display image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_pil.thumbnail((300, 300))
        img_tk = ImageTk.PhotoImage(img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk

    # Browse button
    browse_btn = Button(frame, text='Browse Image', command=lambda: browse_and_predict())
    browse_btn.pack(pady=5)

    def browse_and_predict():
        file_path = filedialog.askopenfilename(title='Select Image', filetypes=[('Images','*.jpg *.jpeg *.png')])
        if file_path:
            predict_from_file(file_path)

    # Dropdown for test_images
    if os.path.isdir(TEST_IMAGES_FOLDER):
        dropdown_frame = Frame(root)
        dropdown_frame.pack(pady=5)
        Label(dropdown_frame, text='Select from test_images').pack()
        files = sorted([f for f in os.listdir(TEST_IMAGES_FOLDER) if f.lower().endswith(('jpg','jpeg','png'))])
        if files:
            var = StringVar(value=files[0])
            menu = OptionMenu(dropdown_frame, var, *files, command=lambda x: predict_from_file(os.path.join(TEST_IMAGES_FOLDER, x)))
            menu.pack()
            # default prediction
            predict_from_file(os.path.join(TEST_IMAGES_FOLDER, files[0]))

    root.mainloop()

# ------------------------------
# Main
# ------------------------------

def main():
    setup_logging()
    # Assume training has been done before launching GUI
    X, y = load_dataset(DATASET_PATH)
    if X.size == 0:
        logging.error('No data loaded, check dataset path and structure.')
        return
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(sss.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_train, y_train)

    launch_interface(pipeline)

if __name__ == '__main__':
    main()
