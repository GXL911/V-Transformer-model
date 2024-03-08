import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

def get_medical_mnist_loaders(data_path, train_batch_size, val_batch_size, test_batch_size):
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loading the dataset
    dataset = datasets.ImageFolder(root=data_path, transform=train_transform)

    # Splitting the dataset into train, validation, and test
    train_indices, test_indices = train_test_split(list(range(len(dataset.targets))), test_size=0.2, stratify=dataset.targets)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, stratify=[dataset.targets[i] for i in train_indices])  # 0.25 x 0.8 = 0.2

    train_data = Subset(dataset, train_indices)
    val_data = Subset(dataset, val_indices)
    test_data = Subset(dataset, test_indices)

    # Data loaders
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def calculate_accuracy(model, data_loader, device, criterion=None):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0
    with torch.no_grad():
        for data in data_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if criterion:
                loss = criterion(outputs, labels)
                running_loss += loss.item()

    accuracy = 100 * correct / total
    loss = running_loss / len(data_loader) if criterion else None
    return loss, accuracy if criterion else accuracy