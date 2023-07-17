import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50


def train_fsl_classifier(dataset_path, num_classes, batch_size, num_epochs):
    # Define transformations for data preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load the MVTec AD dataset
    train_dataset = ImageFolder(dataset_path, transform=transform)

    # Split the dataset into training and validation sets
    num_train_samples = int(0.8 * len(train_dataset))
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [num_train_samples, len(train_dataset) - num_train_samples]
    )

    # Create data loaders for training and validation sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Load the pre-trained ResNet18 model
    model = resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    # Set up the device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Start training
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # Print the loss after each epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Evaluate the model on the validation set
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # Save the trained model
    save_path = "../models/fsl_classifier.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Trained model saved at: {save_path}")


# Example usage
#dataset_path = r"E:\thoucentric\defect_detection\dataset\capsule\test2"
dataset_path = r'E:\thoucentric\defect_detection\dataset\transistor\test'
num_classes = 5  # Number of defect classes in the dataset
batch_size = 32
num_epochs = 10

train_fsl_classifier(dataset_path, num_classes, batch_size, num_epochs)
