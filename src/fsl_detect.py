import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image


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


# Example usage
model_path = "../models/fsl_classifier.pth"
num_classes = 5  # Number of defect classes in the dataset
threshold = 0.9  # Threshold for defect detection probability

model = load_fsl_classifier(model_path, num_classes)

image_path = r"E:\thoucentric\defect_detection\dataset\transistor\test\bent_lead\009.png"
is_defect, predicted_class, probability = detect_defects(image_path, model, threshold)

if is_defect:
    print(f"Defect detected: Class {predicted_class}, Probability: {probability:.4f}")
else:
    print("No defect detected.")
