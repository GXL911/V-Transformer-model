import torch
import torch.nn as nn
import torch.optim as optim
from model import MedicalMNISTCNN  # Updated import
from utils import get_medical_mnist_loaders, calculate_accuracy
def train_model(model, train_loader, val_loader, test_loader, device, epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_batches = len(train_loader)
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Print training progress
            if (i + 1) % 10 == 0:  # Adjust this value as needed
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{total_batches}], Loss: {loss.item():.4f}')

        # Compute average training loss for the epoch
        avg_train_loss = running_loss / total_batches

        # Evaluate on the validation set
        val_loss, val_accuracy = calculate_accuracy(model, val_loader, device, criterion)
        print(f'Epoch {epoch+1} completed - Avg Train Loss: {avg_train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

    # Evaluate on the test set
    test_loss, test_accuracy = calculate_accuracy(model, test_loader, device, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MedicalMNISTCNN().to(device)

    # Update the dataset path accordingly
    data_path = './MedicalMNIST'
    train_loader, val_loader, test_loader = get_medical_mnist_loaders(data_path, train_batch_size=64, val_batch_size=32, test_batch_size=32)

    train_model(model, train_loader, val_loader, test_loader, device, epochs=10, lr=0.001)
