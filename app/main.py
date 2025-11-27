import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("dog_cat_classifier.keras")
    return model

@st.cache_resource
def load_class_names():
    with open("class_names.pkl", "rb") as f:
        class_names = pickle.load(f)
    return class_names

model = load_model()
class_names = load_class_names()


def preprocess(image: Image.Image):
    image = image.resize((224, 224))
    image = image.convert("RGB")
    arr = np.array(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


st.title("Dog vs Cat Classifier")
st.write("Upload an image, and the TensorFlow model will classify it.")

uploaded = st.file_uploader("Upload JPEG or PNG", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    processed = preprocess(img)

    if st.button("Predict"):
        preds = model.predict(processed)
        idx = int(np.argmax(preds[0]))
        label = class_names[idx]
        conf = float(tf.nn.softmax(preds[0])[idx])

        st.subheader("Prediction")
        st.write(f"Class: **{label}**")
        st.write(f"Confidence: **{conf:.4f}**")

