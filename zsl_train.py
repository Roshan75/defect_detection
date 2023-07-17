
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import pickle

# Load dataset and split into training and testing sets
data_path = r"E:\thoucentric\defect_detection\dataset\carpet\test"
attribute_vectors_path = "../models/attribute_vectors.npy"

# Load attribute vectors
attribute_vectors = np.load(attribute_vectors_path)

# Split the dataset into training and testing sets
images = []
labels = []
class_ids = [0, 1, 2, 3, 4, 5]  # Class IDs in your dataset
for class_id in class_ids:
    class_images = np.load(f"{data_path}/{class_id}/images.npy")
    class_labels = np.full((class_images.shape[0],), class_id)
    images.append(class_images)
    labels.append(class_labels)

images = np.concatenate(images)
labels = np.concatenate(labels)
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Extract features using VGG16 model
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

features_train = []
for img_path in images_train:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features_train.append(model.predict(x).flatten())

features_train = np.array(features_train)

features_test = []
for img_path in images_test:
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features_test.append(model.predict(x).flatten())

features_test = np.array(features_test)

# Train ZSL classifier using Logistic Regression
classifier = LogisticRegression(multi_class='ovr')
classifier.fit(features_train, labels_train)

# Predict labels for testing set
predicted_labels = classifier.predict(features_test)

# Evaluate performance
accuracy = accuracy_score(labels_test, predicted_labels)
precision = precision_score(labels_test, predicted_labels, average='macro')
recall = recall_score(labels_test, predicted_labels, average='macro')
f1 = f1_score(labels_test, predicted_labels, average='macro')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Fine-tune and optimize the ZSL classifier as needed.

# Define the parameter grid for hyperparameter optimization
param_grid = {
    'C': [1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Initialize the SVM classifier
svm = SVC()

# Perform hyperparameter optimization using GridSearchCV
grid_search = GridSearchCV(svm, param_grid, cv=5)
grid_search.fit(features_train, labels_train)

# Get the best hyperparameters
best_params = grid_search.best_params_

# Initialize the optimized SVM classifier
best_svm = SVC(**best_params)

# Train the ZSL classifier with the best hyperparameters
best_svm.fit(features_train, labels_train)

# Evaluate performance of the fine-tuned classifier
predicted_labels = best_svm.predict(features_test)

accuracy = accuracy_score(labels_test, predicted_labels)
precision = precision_score(labels_test, predicted_labels, average='macro')
recall = recall_score(labels_test, predicted_labels, average='macro')
f1 = f1_score(labels_test, predicted_labels, average='macro')

with open("../models/best_svm_model.pkl", "wb") as file:
    pickle.dump(best_svm, file)

print("Fine-tuned ZSL Classifier Metrics:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

