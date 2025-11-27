import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import cv2


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("app/../resources/dog_cat_classifier.keras")

@st.cache_resource
def load_class_names():
    with open("app/../resources/class_names.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
class_names = load_class_names()


def preprocess_cv2_style(image: Image.Image):
    # Convert PIL → OpenCV (np array)
    image = np.array(image)

    # Resize like your original script
    image_resized = cv2.resize(image, (224, 224))

    # Scale
    image_scaled = image_resized / 255.0

    # Reshape to (1, 224, 224, 3)
    image_reshaped = np.reshape(image_scaled, (1, 224, 224, 3))

    return image_reshaped


st.title("Dog vs Cat Classifier (Minimalistic)")
st.write("Upload an image and classify it exactly like your Colab predictive system.")

uploaded = st.file_uploader("Upload a cat or dog image", type=["jpg", "jpeg", "png"])

if uploaded:
    # Show uploaded image
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Button to predict
    if st.button("Predict"):
        arr = preprocess_cv2_style(img)

        # Predict using your exact method
        preds = model.predict(arr)
        pred_label = int(np.argmax(preds))

        # Output using your original phrasing
        if pred_label == 0:
            st.subheader("The image is a **cat**.")
        else:
            st.subheader("The image is a **dog**.")
