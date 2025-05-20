# South-African-Bank-Coin-Recognition-System

This document explains how to install dependencies, train the model, and run predictions—including selecting your own images—on the South African coin classification pipeline.

---

## Prerequisites  
- **Python 3.8+**  
- [OpenCV](https://pypi.org/project/opencv-python/) for image processing  
- [scikit-learn](https://pypi.org/project/scikit-learn/) for machine learning  
- [scikit-image](https://pypi.org/project/scikit-image/) for LBP features  
- [NumPy](https://pypi.org/project/numpy/)  

> _Tip:_ Create a virtual environment before installing packages :contentReference[oaicite:0]{index=0}.

---

## Installation  
1. **Clone the repository**  
   ```bash
   git clone https://github.com/yourusername/coin-classifier.git
   cd coin-classifier

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

pip install -r requirements.txt

## requirements.txt should include:
opencv-python
scikit-learn
scikit-image
numpy
