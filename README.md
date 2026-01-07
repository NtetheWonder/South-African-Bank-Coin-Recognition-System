# South-African-Bank-Coin-Recognition-System

This document explains how to install dependencies, train the model, and run predictions—including selecting your own images—on the South African coin classification pipeline.

If you want to a start, to create or, improve on the project  you can clone it or just download the zip file. 

##Or 
You can use the streamlit deployed app if you want to run and test it

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
   git clone https://github.com/NtetheWonder/South-African-Bank-Coin-Recognition-System
   cd South-African-Bank-Coin-Recognition-System

##Creating virtual environment
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate

##Then install
pip install -r requirements.txt

## requirements.txt should include:
opencv-python
scikit-learn
scikit-image
numpy
streamlit
