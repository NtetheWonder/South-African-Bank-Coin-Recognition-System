import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage.feature import local_binary_pattern
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

# Constants for LBP
LBP_RADIUS = 1
LBP_N_POINTS = 8 * LBP_RADIUS
LBP_METHOD = 'uniform'

def preprocess_image(image):
    #Convert image to grayscale and apply Gaussian blur.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imshow("Blur", blur)
    return blur

def segment_coin(image):
    # Segment the coin from the background.
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    
    # Assume the largest contour is the coin
    coin_contour = max(contours, key=cv2.contourArea)
    # Create mask
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [coin_contour], -1, 255, -1)
    return mask

def extract_features(image, mask):
    # Extract Hu Moments, color histogram, and LBP features.
    # Apply mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    # Convert to grayscale
    gray = cv2.cvtColor(masked_image, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("Musked",masked_image)
    # cv2.imshow("Gray", gray)
    # Compute Hu Moments
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments).flatten()
    # Compute color histogram
    hist = cv2.calcHist([masked_image], [0, 1, 2], mask, [8, 8, 8],
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    # Compute LBP
    # gray_float = gray.astype('float') / 255.0
    # gray_normalized = gray/255

    lbp = local_binary_pattern(gray, LBP_N_POINTS, LBP_RADIUS, LBP_METHOD)
    (hist_lbp, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, LBP_N_POINTS + 3),
                                 range=(0, LBP_N_POINTS + 2))
    hist_lbp = hist_lbp.astype("float")
    hist_lbp /= (hist_lbp.sum() + 1e-7)

    # Concatenate features
    features = np.hstack([hu_moments, hist, hist_lbp])
    return features

def load_dataset(dataset_path):
    # Load images and extract features and labels.
    features = []
    labels = []
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if not os.path.isdir(label_path):
            continue
        for filename in os.listdir(label_path):
            file_path = os.path.join(label_path, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue

            preprocessed = preprocess_image(image)
            mask = segment_coin(preprocessed)

            if mask is None:
                continue

            feat = extract_features(image, mask)
            features.append(feat)
            labels.append(label)
    return np.array(features), np.array(labels)

def train_classifier(X, y):
    #Train an SVM classifier.
    model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
   # Evaluate the trained model.
    y_pred = model.predict(X_test)
    all_labels = ['5c','10c','20c','50c','R1','R2','R5']
    print(classification_report(y_test, y_pred, labels= all_labels, target_names= all_labels, zero_division=0))

def predict_image(model, image_path):
    # Predict the class of a single image.
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    preprocessed = preprocess_image(image)
    mask = segment_coin(preprocessed)
    
    if mask is None:
        print("No coin detected.")
        return
    
    feat = extract_features(image, mask).reshape(1, -1)
    prediction = model.predict(feat)[0]
    probability = model.predict_proba(feat).max()
    print(f"Predicted: {prediction} ({probability * 100:.2f}%)")
    # Display the image with prediction

    cv2.putText(image, f"{prediction} ({probability * 100:.2f}%)", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Prediction", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    dataset_path = "dataset"
    test_image_path = "test_images/images1.jpg"  # Replace with your test image
    print("Loading dataset…")
    X, y = load_dataset(dataset_path)
    print(f"Dataset loaded: {len(X)} samples.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Training classifier…")
    model = train_classifier(X_train, y_train)
    print("Evaluating model…")
    evaluate_model(model, X_test, y_test)
    print("Predicting on a new image…")
    predict_image(model, test_image_path)

if __name__ == "__main__":
    main()
