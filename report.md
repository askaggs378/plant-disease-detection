# 🌿 Plant Disease Detection – Final Project Report

## 👋 Introduction

For this project, I built a deep learning model that can identify plant diseases by looking at leaf images. I used a Vision Transformer (ViT) model and a small part of the PlantVillage dataset.

---

## 🧠 What I Did

- I used a pre-trained Vision Transformer model from the `timm` library.
- I picked 5 plant classes from the PlantVillage dataset (3 diseased, 2 healthy).
- I trimmed each class to 100 images to keep training fast.
- I resized, normalized, and augmented the images for better performance.
- I trained the model for 5 epochs and saved it as `outputs/best_model.pth`.

---

## 💻 The Web App

I built a web app using **Streamlit**. It lets users upload a plant leaf image and get a real-time prediction of the disease (or if it’s healthy), along with a confidence score.

You can run it with:

```bash
streamlit run app/app.py
```
---

## ✅ What Worked

- The Vision Transformer performed well even with a small subset of the dataset
- Preprocessing (resizing, normalization, augmentation) helped improve predictions
- The model was able to confidently classify test images
- The Streamlit app worked as expected and gave real-time feedback
- Training was fast and worked well even on a CPU

---

## 🔧 What I'd Improve

- Add more image classes and increase the number of images per class
- Compare the ViT model with a CNN or hybrid model
- Track more evaluation metrics like precision, recall, F1-score, and confusion matrix
- Add the ability to display attention maps or top-3 predictions in the app

---

## 📽️ Video Demo

[Click here to watch the demo](https://www.loom.com/share/f050790a13124a01bd2c0328d6838113?sid=97e82029-8ce9-4767-a8ff-f1e0aaa31e47)

## 🔗 GitHub Repository

You can view the full project and code here:  
[https://github.com/askaggs378/plant-disease-detection](https://github.com/askaggs378/plant-disease-detection)
