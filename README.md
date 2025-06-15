# 🍎 Fruits and Vegetables Image Recognition using CNN

This project is a deep learning-based image classification system that identifies different fruits and vegetables using Convolutional Neural Networks (CNNs). It is built using Python, TensorFlow, and Keras, and is capable of recognizing images in real-time with high accuracy.

---

## 📌 Project Overview

Manual classification of fruits and vegetables in domains like retail, agriculture, and dietary monitoring is error-prone and inefficient. This project automates the process using a trained CNN model, providing fast and accurate predictions to support smart applications.

---

## 🚀 Features

- Classifies 15+ types of fruits and vegetables
- Uses Convolutional Neural Networks (CNN) for classification
- Trained on 15,000+ labeled images
- Real-time prediction support
- Data augmentation to prevent overfitting
- Ready for deployment in mobile/web apps

---

## 🧠 Technologies Used

- Python 3.7+
- TensorFlow & Keras
- OpenCV
- NumPy
- Matplotlib
- Google Colab / Jupyter Notebook

---

## 🗂️ Dataset

- **Source**: [Kaggle - Fruits and Vegetables Image Recognition](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition)
- Over 15,000 labeled images of fruits and vegetables
- Includes common classes like Apple, Banana, Tomato, Potato, Carrot, etc.
- Preprocessing steps:
  - Resizing images to 100x100 pixels
  - Normalizing pixel values to range [0,1]
  - Applying data augmentation (rotation, flipping, zooming)

---

## 📁 Folder Structure

  ````markdown
  ├── trainFruit.ipynb     # Model training notebook  
  ├── test.ipynb           # Prediction and testing notebook  
  ├── dataset/             # Training/testing image data  
  ├── saved_model/         # Trained model files  
  ├── images/              # Sample images for testing  
  └── README.md            # Project documentation  
  ````


---

## 🛠️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/fruit-veggie-classifier.git
   cd fruit-veggie-classifier
   
2. **Install dependencies**
   ```bash
   pip install -r requirements.txt

3. **Train the model**
- Open trainFruit.ipynb in Jupyter or Colab and run all cells.

4. **Test the model**
- Use test.ipynb to load the saved model and make predictions on sample images.

---

## 📊 Results

| **Metric**         | **Score**    |
|--------------------|--------------|
| Training Acc.      | ~98%         |
| Validation Acc.    | ~93–95%      |
| Test Acc.          | ~92%         |

- Confusion matrix used to analyze misclassifications
- Misclassification mostly occurred in visually similar classes (e.g., Potato vs. Ginger)
- Data augmentation helped in improving generalization

---

## 📱 Future Scope

- Expand dataset to include more classes and varied lighting/backgrounds  
- Implement transfer learning using MobileNet, ResNet, or EfficientNet  
- Deploy as a web app using Flask or mobile app using TensorFlow Lite  
- Add audio output (text-to-speech) for accessibility  

---
