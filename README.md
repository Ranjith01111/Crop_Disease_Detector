# 🍅 AgriMind: AI-Based Crop Disease Detector

**AgriMind** is an AI-powered system designed to detect plant diseases from leaf images. Built using a Convolutional Neural Network (CNN) in TensorFlow and an interactive web dashboard in Streamlit, this tool aims to provide quick and accurate disease diagnosis to help maintain crop health.

Currently, the model is configured to detect **Early Blight** versus **Healthy** tomato leaves, achieving high accuracy on the test set.

---

## 🚀 Features

- **AI Disease Diagnosis:** Uses a custom-built Convolutional Neural Network (CNN) to detect diseases from leaf images.
- **Interactive UI:** A highly interactive, easy-to-use web interface built with Streamlit.
- **Real-time Reporting:** Provides instant predictions, confidence scores, and recommended remedies.
- **Scan History:** Keeps track of your recent image scans during the session.

---

## 🛠️ Technology Stack

- **Machine Learning & AI:** TensorFlow, Keras
- **Web App Framework:** Streamlit
- **Image Processing:** Pillow (PIL), NumPy
- **Language:** Python

---

## 📂 Project Structure

- `My_Project/app.py`: The machine learning training script. It loads the dataset, builds the CNN architecture, trains the model, and saves it as `tomato_disease_model.h5`.
- `My_Project/predict.py`: The Streamlit web application. It loads the trained `.h5` model, provides the user interface for uploading images, and calculates predictions.
- `tomato_disease_model.h5`: The trained weights and architecture of the neural network (generated after running `app.py`).

---

## 💻 How to Run Locally

### 1. Prerequisites
Ensure you have Python installed on your system. It is highly recommended to use a virtual environment.

Install the required dependencies:
```bash
pip install tensorflow streamlit numpy pillow
```

### 2. Train the Model (Optional)
If you want to train the model yourself, ensure you have your dataset in folders named `train` and `test` inside the project root, then run:
```bash
python My_Project/app.py
```
*(Note: A pre-trained model might already be available in your working directory.)*

### 3. Run the Web App
To start the Streamlit dashboard, run the following command in your terminal:
```bash
streamlit run My_Project/predict.py
```

The app will automatically open in your default web browser at `http://localhost:8501`.

---

## 🤝 Contributing
Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

---
*Developed by Ranjith V (Smacky's Bro) | Mini Project Semester 6*
