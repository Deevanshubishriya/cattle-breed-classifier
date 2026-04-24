# 🐄 Cattle Breed Classifier

A deep learning project that uses a Convolutional Neural Network (MobileNetV2) to identify the breed of cattle from an uploaded image. 

**🔴 Live Demo:** [Click here to try the Live App on Hugging Face](https://huggingface.co/spaces/Deevanshu2109/cattle-breed-classifier)

## 📖 Overview
This project was built to automate the identification of various cattle breeds using Computer Vision. The model was trained using a custom dataset of cattle images and achieved good accuracy using Transfer Learning. A user-friendly web interface is provided via Gradio, making it easy for anyone to upload a photo and get a prediction instantly.

## ✨ Features
* Image Classification: Identifies up to 15 different cattle breeds.
* Transfer Learning: Built on top of the highly efficient `MobileNetV2` architecture.
* Web Interface: Interactive UI built with Gradio.
* Lightweight & Fast: Deployed using `tensorflow-cpu` for quick inference on standard hardware.

## 🗂️ Dataset
The model was trained on a custom dataset containing 3,750 images spread across 15 cattle breed classes. The dataset was split into 80% training and 20% validation sets.

## 🛠️ Tech Stack
* Deep Learning Framework: TensorFlow & Keras
* Pre-trained Model: MobileNetV2
* Web Interface: Gradio
* Deployment: Hugging Face Spaces

## 🚀 How to Run Locally
1. Clone the repository:
   ```bash
   git clone [https://github.com/Deevanshu2109/cattle-breed-classifier.git](https://github.com/Deevanshu2109/cattle-breed-classifier.git)
   cd cattle-breed-classifier
2.Install requirements:
   ```bash
   pip install -r requirements.txt
