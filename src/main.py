################# VERSION 1 #############################
# import os
# import cv2
# import numpy as np
# from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import train_test_split
#
# # Path to the MVTec AD dataset images
# dataset_path = '../capsule'
#
# # Load the attributes and corresponding class labels from the attribute file
# attribute_file_path = '../attributes.txt'
# attributes = []
# class_labels = []
# with open(attribute_file_path, 'r') as attribute_file:
#     for line in attribute_file:
#         line = line.strip().split('\t')
#         class_labels.append(line[0])
#         attributes.append([float(attr) for attr in line[1:]])
#
# # Convert attributes and class labels to numpy arrays
# attributes = np.array(attributes)
# class_labels = np.array(class_labels)
#
# # Split the data into training and test sets
# X_train_attr, X_test_attr, y_train, y_test = train_test_split(attributes, class_labels, test_size=0.2, random_state=42)
#
# # Load the pre-trained VGG16 model
# vgg_model = VGG16(weights='imagenet', include_top=False)
#
# # Function to extract image features using VGG16
# def extract_image_features(image_path):
#     # Load and preprocess the image
#     image = cv2.imread(image_path)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     image = cv2.resize(image, (224, 224))  # VGG16 input size
#     image = preprocess_input(image)
#
#     # Reshape the image to match VGG16 input shape
#     image = np.expand_dims(image, axis=0)
#
#     # Extract features using VGG16
#     features = vgg_model.predict(image)
#     features = features.flatten()
#
#     return features
#
# # Extract image features for training set
# X_train_img = []
# for image_path in y_train:
#     class_dir = os.path.join(dataset_path, image_path)
#     if os.path.isdir(class_dir):
#         for image_name in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, image_name)
#             img_features = extract_image_features(img_path)
#             X_train_img.append(img_features)
#             break
# X_train_img = np.array(X_train_img)
#
# # Extract image features for test set
# X_test_img = []
# for image_path in y_test:
#     class_dir = os.path.join(dataset_path, image_path)
#     if os.path.isdir(class_dir):
#         for image_name in os.listdir(class_dir):
#             img_path = os.path.join(class_dir, image_name)
#             img_features = extract_image_features(img_path)
#             #img_features = extract_image_features(os.path.join(dataset_path, image_path))
#             X_test_img.append(img_features)
#             break
# X_test_img = np.array(X_test_img)
#
# # Concatenate image and attribute features
# # X_train = np.concatenate((X_train_img, X_train_attr), axis=1)
# # X_test = np.concatenate((X_test_img, X_test_attr), axis=1)
#
# # # Scale the features
# # scaler = StandardScaler()
# # X_train = scaler.fit_transform(X_train)
# # X_test = scaler.transform(X_test)
# #
# # # Train a classifier using the training set
# # classifier = SVC(kernel='linear')
# # classifier.fit(X_train, y_train)
# #
# # # Predict class labels for the test set
# # y_pred = classifier.predict(X_test)
# #
# # # Compute accuracy
# # accuracy = accuracy_score(y_test, y_pred)
# # print("Accuracy:", accuracy)
#
# # Scale the image features
# scaler = StandardScaler()
# X_train_img = scaler.fit_transform(X_train_img)
# X_test_img = scaler.transform(X_test_img)
#
# # Scale the attribute features
# scaler_attr = StandardScaler()
# X_train_attr = scaler_attr.fit_transform(X_train_attr)
# X_test_attr = scaler_attr.transform(X_test_attr)
#
# # Concatenate image and attribute features
# X_train = np.concatenate((X_train_img, X_train_attr), axis=1)
# X_test = np.concatenate((X_test_img, X_test_attr), axis=1)
#
# # Train a classifier using the training set
# classifier = SVC(kernel='linear')
# classifier.fit(X_train, y_train)
#
# # Predict class labels for the test set
# y_pred = classifier.predict(X_test)
#
# # Compute accuracy
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)

#############  VERSION 2  ################################
# import os
# import numpy as np
# from attributes import image_path_generator
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# import pickle
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
#
# ## Load dataset and split into training and testing sets
# data_path = r"E:\thoucentric\defect_detection\dataset\carpet\test"
# attribute_vectors_path = "attribute_vectors.npy"
# class_names = ['color', 'cut', 'good', 'hole', 'metal_contamination', 'thread']
# attribute_vectors = np.load(attribute_vectors_path)
#
# # Split the dataset into training and testing sets
# images = []
# labels = []
# for class_id in range(attribute_vectors.shape[0]):
#     try:
#         class_images = np.load(f"{data_path}/{str(class_id)}/images.npy")
#     except:
#         image_path_generator(data_path)
#         class_images = np.load(f"{data_path}/{str(class_id)}/images.npy")
#     class_labels = np.full((class_images.shape[0], 1), class_id)
#     #class_labels = [str(class_id)]
#     images.append(class_images)
#     labels.append(class_labels)
#
# images = np.concatenate(images)
# labels = np.concatenate(labels)
# images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)
#
# ## Extract features using VGG16 model
# base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
#
# features_train = []
# for img_path in images_train:
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features_train.append(model.predict(x).flatten())
#
# features_train = np.array(features_train)
#
# features_test = []
# for img_path in images_test:
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features_test.append(model.predict(x).flatten())
#
# features_test = np.array(features_test)
#
# ## Train ZSL classifier using Linear Regression
# regressor = LinearRegression()
# regressor.fit(attribute_vectors[labels_train.flatten()], features_train)
#
# ## Predict features for testing set
# predicted_features = regressor.predict(attribute_vectors[labels_test.flatten()])
# print(predicted_features)
#
# ## Evaluate performance
#
# predicted_labels = np.argmax(predicted_features, axis=1)
#
# # Calculate evaluation metrics
# accuracy = accuracy_score(labels_test.flatten(), predicted_labels)
# precision = precision_score(labels_test.flatten(), predicted_labels, average='macro')
# recall = recall_score(labels_test.flatten(), predicted_labels, average='macro')
# f1 = f1_score(labels_test.flatten(), predicted_labels, average='macro')
#
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
#
# svm = SVC()
#
# # Define the parameter grid for hyperparameter optimization
# param_grid = {
#     'C': [1, 10, 100],
#     'gamma': [0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear']
# }
#
# # Perform hyperparameter optimization using grid search
# grid_search = GridSearchCV(svm, param_grid, cv=5)
# grid_search.fit(attribute_vectors, labels_train.flatten())
#
# # Retrieve the best hyperparameters
# best_params = grid_search.best_params_
#
# # Reinitialize the SVM with the best hyperparameters
# best_svm = SVC(**best_params)
#
# # Train the ZSL classifier with the best hyperparameters
# best_svm.fit(attribute_vectors, labels_train.flatten())
#
# # Predict features for the testing set using the optimized classifier
# predicted_features = best_svm.predict(attribute_vectors[labels_test.flatten()])
#
# # Evaluate performance with the optimized classifier
# accuracy = accuracy_score(labels_test.flatten(), predicted_features)
# precision = precision_score(labels_test.flatten(), predicted_features, average='macro')
# recall = recall_score(labels_test.flatten(), predicted_features, average='macro')
# f1 = f1_score(labels_test.flatten(), predicted_features, average='macro')
#
# # Load the saved best_svm model
# with open("best_svm_model.pkl", "wb") as file:
#     pickle.dump(best_svm)
#
# print("Optimized ZSL Classifier Metrics:")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
#
#
#
# #############  VERSION 3  #################################
# import numpy as np
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.models import Model
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.svm import SVC
# import pickle
#
# # Step 1: Load dataset and split into training and testing sets
# data_path = r"E:\thoucentric\defect_detection\dataset\carpet\test"
# attribute_vectors_path = "attribute_vectors.npy"
#
# # Load attribute vectors
# attribute_vectors = np.load(attribute_vectors_path)
#
# # Split the dataset into training and testing sets
# images = []
# labels = []
# class_ids = [0, 1, 2, 3, 4, 5]  # Class IDs in your dataset
# for class_id in class_ids:
#     class_images = np.load(f"{data_path}/{class_id}/images.npy")
#     class_labels = np.full((class_images.shape[0],), class_id)
#     images.append(class_images)
#     labels.append(class_labels)
#
# images = np.concatenate(images)
# labels = np.concatenate(labels)
# images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.2, random_state=42)
#
# # Step 2: Extract features using VGG16 model
# base_model = VGG16(weights='imagenet')
# model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)
#
# features_train = []
# for img_path in images_train:
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features_train.append(model.predict(x).flatten())
#
# features_train = np.array(features_train)
#
# features_test = []
# for img_path in images_test:
#     img = image.load_img(img_path, target_size=(224, 224))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     features_test.append(model.predict(x).flatten())
#
# features_test = np.array(features_test)
#
# # Step 3: Train ZSL classifier using Logistic Regression
# classifier = LogisticRegression(multi_class='ovr')
# classifier.fit(features_train, labels_train)
#
# # Step 4: Predict labels for testing set
# predicted_labels = classifier.predict(features_test)
#
# # Step 5: Evaluate performance
# accuracy = accuracy_score(labels_test, predicted_labels)
# precision = precision_score(labels_test, predicted_labels, average='macro')
# recall = recall_score(labels_test, predicted_labels, average='macro')
# f1 = f1_score(labels_test, predicted_labels, average='macro')
#
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)
#
# # Step 6: Fine-tune and optimize the ZSL classifier as needed.
#
# # Define the parameter grid for hyperparameter optimization
# param_grid = {
#     'C': [1, 10, 100],
#     'gamma': [0.1, 0.01, 0.001],
#     'kernel': ['rbf', 'linear']
# }
#
# # Initialize the SVM classifier
# svm = SVC()
#
# # Perform hyperparameter optimization using GridSearchCV
# grid_search = GridSearchCV(svm, param_grid, cv=5)
# grid_search.fit(features_train, labels_train)
#
# # Get the best hyperparameters
# best_params = grid_search.best_params_
#
# # Initialize the optimized SVM classifier
# best_svm = SVC(**best_params)
#
# # Train the ZSL classifier with the best hyperparameters
# best_svm.fit(features_train, labels_train)
#
# # Step 7: Evaluate performance of the fine-tuned classifier
# predicted_labels = best_svm.predict(features_test)
#
# accuracy = accuracy_score(labels_test, predicted_labels)
# precision = precision_score(labels_test, predicted_labels, average='macro')
# recall = recall_score(labels_test, predicted_labels, average='macro')
# f1 = f1_score(labels_test, predicted_labels, average='macro')
#
# with open("best_svm_model.pkl", "wb") as file:
#     pickle.dump(best_svm)
#
# print("Fine-tuned ZSL Classifier Metrics:")
# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)