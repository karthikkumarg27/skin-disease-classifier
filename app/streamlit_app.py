# streamlit_app.py - Web interface for model prediction

import streamlit as st
import numpy as np
import sys
import os

# 👉 Add parent directory to path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from src.evaluate import get_gradcam_heatmap
import cv2

st.title("🩺 Skin Disease Classifier")

model = load_model("final_model.h5")
class_names = ['Actinic keratoses', 'Basal cell carcinoma', 'Benign keratosis',
               'Dermatofibroma', 'Melanoma', 'Melanocytic nevi', 'Vascular lesions']

uploaded = st.file_uploader("Upload a skin image", type=["jpg", "png"])

if uploaded:
    img = image.load_img(uploaded, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    label = class_names[np.argmax(pred)]

    st.image(img, caption=f"Predicted: {label}", width=300)

    # Grad-CAM
    heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name="top_conv")
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    st.image(heatmap, caption="Grad-CAM Heatmap", width=300)
