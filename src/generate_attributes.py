import os
import cv2
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input

# Path to the MVTec AD dataset images
dataset_path = r'../capsule'

# Load the pre-trained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Function to extract attributes from an image using VGG16
def extract_attributes(image_path):
    # Load and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))  # VGG16 input size
    image = preprocess_input(image)

    # Reshape the image to match VGG16 input shape
    image = np.expand_dims(image, axis=0)

    # Extract features using VGG16
    features = model.predict(image)
    features = features.flatten()

    return features

# Iterate through the images in the MVTec AD dataset
#class_attr = dict()
keys = []
values = []
for class_name in os.listdir(dataset_path):
    class_dir = os.path.join(dataset_path, class_name)
    if os.path.isdir(class_dir):
        #for image_name in os.listdir(class_dir):
        image_path = os.path.join(class_dir, os.listdir(class_dir)[0])

        # Extract attributes from the image
        attributes = extract_attributes(image_path)

        # Process and store the attributes as needed
        # ...

        # Print the attributes for verification
        print("Class:", class_name)
        #print("Image:", image_name)
        print("Attributes:", attributes)
        print("-" * 30)
        keys.append(class_name)
        values.append(attributes.tolist())
        #class_attr = dict(zip(class_name, attributes.tolist()))
class_attributes = {k: v for k, v in zip(keys, values)}
#print(attributes)
attribute_file_path = '../attributes.txt'
with open(attribute_file_path, 'w') as attribute_file:
    for class_name in keys:
        # Get the corresponding attributes for the class
        attributes = class_attributes.get(class_name, [])

        # Write class name and attributes to the file
        attribute_file.write(class_name + '\t')
        attribute_file.write('\t'.join(str(attr) for attr in attributes))
        attribute_file.write('\n')

# Print the contents of the attribute file
with open(attribute_file_path, 'r') as attribute_file:
    print(attribute_file.read())