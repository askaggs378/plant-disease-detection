# 🌿 Plant Disease Detection

This project uses a deep learning model (Vision Transformer) to detect plant leaf diseases using images. It includes a simple web app built with Streamlit where you can upload a photo and get a prediction.

---

## 🔍 What It Does

- Trains a model on leaf images from the PlantVillage dataset
- Uses a Vision Transformer (ViT) from the `timm` library
- Allows users to upload a leaf photo and receive a disease prediction with confidence

---

## 🧠 Dataset and Model

- Dataset: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
- Classes used (trimmed to ~100 images each for faster training):
  - Tomato___Early_blight
  - Tomato___Late_blight
  - Tomato___healthy
  - Potato___healthy
  - Pepper__bell___Bacterial_spot

- Model: Vision Transformer (`vit_base_patch16_224`)

---


## 📁 Project Structure

- `app/` – Streamlit app (app.py)
- `data/` – Subset of the PlantVillage dataset used for training
- `models/` – Model architecture setup (Vision Transformer)
- `utils/` – Preprocessing code (resizing, augmentation, etc.)
- `outputs/` – Trained model output file (.pth)
- `main.py` – Script to train the model
- `trim_subset.py` – Helper script to reduce images per class
- `README.md` – You're reading it!

---

## 🚀 How to Use It

### 1. Clone the repo and install the dependencies

```
git clone https://github.com/askaggs378/plant-disease-detection.git
cd plant-disease-detection
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧪 Train the Model (Optional)

I trained a Vision Transformer (ViT) on a subset of the PlantVillage dataset using five classes. Each class was trimmed to around 100 images for faster training.

To re-run the training process:

```
python main.py
```


This will:
- Train the model for 5 epochs
- Save the weights to:
  `outputs/best_model.pth`


---

## 💻 Run the Web App

To launch the Streamlit app:

```
streamlit run app/app.py
```

Then open your browser and go to:

```
http://localhost:8501
```

Upload a `.jpg` or `.png` image of a plant leaf.  
The app will display:
- The predicted disease class
- The confidence score

