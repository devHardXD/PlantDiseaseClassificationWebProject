import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import tensorflow as tf
import catboost
from PIL import Image

# Load Models
model_extractor = tf.keras.models.load_model('feature_extractor.keras')
model_classification = catboost.CatBoostClassifier()
model_classification.load_model('feature_classifier.cbm')

# Class Names
class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

# Define the prediction function
def predict_image(file, threshold=0.5):
    img = Image.open(file).convert('RGB')  # Convert image to RGB to ensure 3 channels
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # Feature Extraction
    features = model_extractor.predict(img_array)

    # Classification
    predictions = model_classification.predict_proba(features)

    if predictions.shape[1] == 4:
        predicted_labels = predictions > threshold
    else:
        raise ValueError(f"Mismatch between number of predictions ({predictions.shape[1]}) and expected classes (4).")

    return img, predicted_labels, predictions

# Streamlit App
st.title("Image Prediction for Disease Classification")

st.write("Upload an image to predict its class (Black Rot, ESCA, Healthy, or Leaf Blight).")

# Image Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Predict Image if uploaded
    img, predicted_labels, predictions = predict_image(uploaded_file, threshold=0.5)

    # Get predicted class names (if any are above threshold)
    predicted_classes = [class_names[i] for i in range(len(class_names)) if predicted_labels[0][i]]

    # Search for Max Pred
    max_index = np.argmax(predictions[0])
    max_class = class_names[max_index]
    max_value = predictions[0][max_index]

    # Display the image with predictions
    st.image(img, caption=f"Uploaded Image", use_column_width=True)

    # Display Prediction
    if predicted_classes:
        st.write(f"Predicted: {', '.join(predicted_classes)}")
        st.write(f"Confidence: {max_value:.2f}")
    else:
        st.write("No classes predicted above the threshold.")
