import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.applications.efficientnet import preprocess_input
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import catboost

# Load Model
model_extractor = tf.keras.models.load_model('feature_extractor.keras')
model_classification = catboost.CatBoostClassifier()
model_classification.load_model('feature_classifier.cbm')

# Load and predict Image
def predict_image(filepath, threshold=0.5):
    img = image.load_img(filepath, target_size=(224, 224))
    img = preprocess_input(img)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)

    # Feature Extractor
    features = model_extractor.predict(img_array)

    # Feature Classification
    predictions = model_classification.predict_proba(features)

    # Conditional for predictions (Debugger)
    if predictions.shape[1] == 4:
        predicted_labels = predictions > threshold
    else:
        raise ValueError(f"Mismatch between number of predictions ({predictions.shape[1]}) and expected classes (4).")

    return img, predicted_labels, predictions

# Open Image using TKDialog
def upload_images():
    root = tk.Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(title="Select images", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    return root.tk.splitlist(file_paths)

# Main code
uploaded_files = upload_images()

# Class Name
class_names = ['Black Rot', 'ESCA', 'Healthy', 'Leaf Blight']

for file_path in uploaded_files:
    img, predicted_labels, predictions = predict_image(file_path, threshold=0.5)

    # Get predicted class names (if any are above threshold)
    predicted_classes = [class_names[i] for i in range(len(class_names)) if predicted_labels[0][i]]

    # Search for Max Pred
    max_index = np.argmax(predictions[0])
    max_class = class_names[max_index]
    max_value = predictions[0][max_index]

    # Display the Image with Probability and Class Predicted
    plt.imshow(img)
    if predicted_classes:
        title = f"Predicted: {', '.join(predicted_classes)} (Confidence: {max_value:.2f})"
    else:
        title = "Predicted: None"
    plt.title(title)
    plt.axis('off')
    plt.show()
