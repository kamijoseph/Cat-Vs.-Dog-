# **README.md — Dog vs Cat Classifier using MobileNetV2 + Streamlit**

## **1. Overview**

This project implements an end-to-end **Dog vs Cat Image Classification** system using **Transfer Learning with MobileNetV2** and a lightweight **Streamlit web interface** for real-time prediction. The objective is to provide a fast, efficient, and accurate model capable of distinguishing images of dogs and cats with minimal compute requirements.

The project follows a clean pattern:

1. **Data Preparation** – Using a curated subset of the popular Kaggle “Dogs vs Cats” dataset (2000 dog images + 2000 cat images).
2. **Model Training** – Fine-tuning a pretrained **MobileNetV2** feature extractor from `tf.keras.applications`.
3. **Model Serialization** – Saving the trained network using the modern `.keras` format and storing class names separately in a `.pkl` file.
4. **Deployment** – Building a Streamlit application that loads the model, preprocesses uploaded images with OpenCV-style transforms, and outputs predictions interactively.

This repository includes all necessary assets and instructions to train the model, reproduce results, and deploy locally or on cloud infrastructure.

---

## **2. Problem Statement**

Classifying animals—particularly dogs and cats—poses a simple yet instructive visual recognition challenge. Traditional computer vision struggles with the high variability in:

* breed types
* lighting
* image backgrounds
* angles and poses

Using deep learning, especially with pretrained models, greatly improves accuracy and inference speed.

The goal of this project:

> **Given an input image, determine whether it contains a dog or a cat.**

This is a **binary classification** problem with mutually exclusive classes. The project emphasizes reproducibility, simplicity, and modern TensorFlow practices.

---

## **3. Dataset**

This project uses **4,000 images** sampled from the well-known Kaggle Dogs vs Cats dataset:

* **2,000 cat images**
* **2,000 dog images**

Dataset Reference (Kaggle):
[https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)

All images were resized to **224×224×3** during preprocessing to match MobileNetV2 input specifications.

---

## **4. Model Architecture**

The heart of the system is **MobileNetV2**, a lightweight convolutional neural network optimized for mobile and edge devices. It is loaded directly from **TensorFlow Keras Applications** with pretrained **ImageNet** weights.

### **Key Architecture Choices**

* **Base Model:** `MobileNetV2(include_top=False, weights="imagenet")`
* **Input Size:** 224 × 224 × 3
* **Trainable Layers:** Last convolutional block fine-tuned
* **Additional Layers:**

  * GlobalAveragePooling2D
  * Dense(128, activation='relu')
  * Dropout(0.2)
  * Dense(2, activation='softmax')

### **Why MobileNetV2?**

* Highly efficient for CPU inference
* Excellent performance with small datasets
* Built-in depthwise separable convolutions reduce parameters
* Robust generalization after fine-tuning

This results in a fast, accurate classifier suitable for real-time prediction in a Streamlit app.

---

## **5. Training Pipeline**

The model was trained using:

* **Optimizer:** Adam (learning rate tuned during experiments)
* **Loss Function:** Sparse Categorical Crossentropy
* **Metrics:** Accuracy
* **Epochs:** 10–20 depending on training stability
* **Data Augmentation:**

  * Random flips
  * Random rotations
  * Rescaling

Training stabilized quickly due to transfer learning and ImageNet initialization.

### **Outputs:**

* **dog_cat_classifier.keras** — Saved using the modern `.keras` format
* **class_names.pkl** — Contains:

  ```python
  {0: "cat", 1: "dog"}
  ```

---

## **6. Streamlit Application**

The deployed interface provides a simple user experience:

* Upload an image (`jpg`, `jpeg`, `png`)
* Model automatically resizes to **224×224**
* Preprocesses with OpenCV-style pipeline
* Runs prediction using the trained MobileNetV2 network
* Outputs the predicted class label

### **Main Features**

* Lightweight UI
* Caches model load using `st.cache_resource`
* Handles PIL images, converts to NumPy, and feeds into TensorFlow
* Accurate prediction probability from the softmax layer

---


## **7. Preprocessing Flow**

The Streamlit app uses a simple OpenCV-style pipeline:

1. Convert PIL → NumPy
2. Resize to 224×224
3. Scale pixel values to `[0,1]`
4. Reshape to `(1, 224, 224, 3)`
5. Perform inference

This ensures consistency with the training pipeline.

---
