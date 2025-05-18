import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
import sys

# Import path 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.vit_model import build_vit

# Path to saved model
MODEL_PATH = "outputs/best_model.pth"
DATA_DIR = "data/Subset"

# Load class names
class_names = sorted(os.listdir(DATA_DIR))
num_classes = len(class_names)

# Load model
@st.cache_resource
def load_model():
    model = build_vit(num_classes=num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Streamlit app layout
st.title("🌿 Plant Disease Detector")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

    # Preprocess the image
    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        class_name = class_names[predicted_class]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

    st.success(f"Prediction: **{class_name}** ({confidence * 100:.2f}% confidence)")

