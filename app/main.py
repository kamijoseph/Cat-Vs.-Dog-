import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import cv2


# resoures
@st.cache_resource
def load_dog_cat_model():
    return tf.keras.models.load_model("app/../resources/dog_cat_classifier.keras")


@st.cache_resource
def load_dog_cat_class_names():
    with open("app/../resources/class_names.pkl", "rb") as f:
        return pickle.load(f)


# Load model + class names
dog_cat_model = load_dog_cat_model()
dog_cat_classes = load_dog_cat_class_names() 


#preprocessing

def preprocess_cv2_style(image: Image.Image, size=(224, 224)):
    image = np.array(image)
    img_resized = cv2.resize(image, size)
    img_scaled = img_resized / 255.0
    img_reshaped = np.reshape(img_scaled, (1, size[0], size[1], 3))
    return img_reshaped


# st ui

st.title("Visual Intelligence Hub")
st.write(
    "A unified interface for computer vision tasks. "
    "Currently enabled: **Dog vs Cat Classification**."
)

st.divider()

st.header("Dog vs Cat Classifier")
st.write("Upload an image and classify it using your pretrained dog/cat model.")

uploaded = st.file_uploader("Upload a dog or cat image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        arr = preprocess_cv2_style(img, size=(224, 224))
        preds = dog_cat_model.predict(arr)
        pred_label = int(np.argmax(preds))
        st.subheader(f"Prediction: **{dog_cat_classes[pred_label]}**")
