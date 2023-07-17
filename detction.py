'''

Run detection code in terminal "python detection.py --input_image path_to_input_image --model (fsl or zsl)
--model default will be zsl

'''

from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16
from tensorflow.keras.models import Model
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

flags.DEFINE_string('input_image', None, 'path to input image')
flags.DEFINE_string('model', "zsl", 'Select zsl or fsl')

def zsl_test(img):
    with open("models/best_svm_model.pkl", "rb") as file:
        best_svm = pickle.load(file)

    # Load the pre-trained VGG16 model for feature extraction
    base_model = VGG16(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    class_names = ['color', 'cut', 'good', 'hole', 'metal_contamination', 'unseen']

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


def load_fsl_classifier(model_path, num_classes):
    # Load the pre-trained ResNet18 model
    model = resnet50(pretrained=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    return model


def detect_defects(image_path, model, threshold):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # Make a prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        _, predicted_class = torch.max(outputs, 1)

    # Check if the predicted class is a defect
    if probabilities[predicted_class] > threshold:
        return True, predicted_class.item(), probabilities[predicted_class].item()
    else:
        return False, predicted_class.item(), probabilities[predicted_class].item()


def fsl_test(image_path):
    model_path = "models/fsl_classifier.pth"
    num_classes = 5  # Number of defect classes in the dataset
    threshold = 0.9  # Threshold for defect detection probability

    model = load_fsl_classifier(model_path, num_classes)

    #image_path = r"E:\thoucentric\defect_detection\dataset\transistor\test\bent_lead\009.png"
    is_defect, predicted_class, probability = detect_defects(image_path, model, threshold)

    if is_defect:
        print(f"Defect detected: Class {predicted_class}, Probability: {probability:.4f}")
    else:
        print("No defect detected.")

def main(_argv):
    print(FLAGS.input_image)
    print(FLAGS.model)
    try:
        img = cv2.imread(FLAGS.input_image)
        img = cv2.resize(img,(224,224))
    except:
        raise Exception("Please select valid image with '--input_image'")

    if FLAGS.model == 'fsl':
        fsl_test(FLAGS.input_image)
    elif FLAGS.model == 'zsl':
        zsl_test(img)
    else:
        raise Exception("Please select model either zsl or fsl")



if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass