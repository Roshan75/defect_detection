import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
import pickle

# Load the saved best_svm model
with open("../models/best_svm_model.pkl", "rb") as file:
    best_svm = pickle.load(file)

# Load the pre-trained VGG16 model for feature extraction
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

class_names = ['color', 'cut', 'good', 'hole', 'metal_contamination', 'unseen']

# Load the image for object detection
image_path = r"E:\thoucentric\defect_detection\dataset\carpet\test\4\004.png"  # Replace with the path to your image
img = cv2.imread(image_path)
img = cv2.resize(img, (224, 224))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess the image
img_preprocessed = image.img_to_array(img)
img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
img_preprocessed = preprocess_input(img_preprocessed)

# Extract features using VGG16 model
features = model.predict(img_preprocessed).flatten()

# Use the best_svm model for object detection
predicted_label = best_svm.predict([features])[0]

# Print the predicted label
print("Predicted Index:", predicted_label)
print("Predicted Class:", class_names[int(predicted_label)])