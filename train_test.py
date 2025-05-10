import argparse
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from model import SmokingCNN
from dataset import SmokingDataset

def calculate_metrics(y_true, y_pred):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred)
    metrics['recall'] = recall_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred)

    return metrics

def main(data_path, version, epochs, learning_rate):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA available...")
    else:
        device = torch.device("cpu")
        print("CUDA is not available...")
    
    dataset = SmokingDataset(data_path)

    train_dataset, test_dataset = train_test_split(dataset, test_size=0.2)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = SmokingCNN(version=version)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Train model...")
    best_val_accuracy = 0.0
    best_model_path = './saves/best_model.pth'

    for epoch in range(epochs):
        model.train()
        
        train_loss = 0
        train_correct = 0
        train_total = 0

        # Обучение
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (outputs > 0.5).float()
            train_correct += (predictions == labels).sum().item()
            train_total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # Валидация
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train: Loss - {avg_train_loss:.4f}, Accuracy - {train_accuracy:.4f}")
        print(f"Valid: Loss - {avg_val_loss:.4f}, Accuracy - {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), best_model_path)

    # Тестирование на лучшей модели
    print("Testing model...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend((outputs.cpu().numpy() > 0.5).astype(int)) 

    metrics = calculate_metrics(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print("Test Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print("Confusion Matrix:")
    print(cm)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test a neural network model.')
    parser.add_argument('--data_path', type=str, help='Path to the JSON data file.')
    parser.add_argument('--version', type=int, help='Version of CNN model.')
    parser.add_argument('--epochs', type=int, help='Number of training epochs.')
    parser.add_argument('--learning_rate', type=float, help='Number of learning rate', default=0.01)

    args = parser.parse_args()

    start_time = time.time()
    main(args.data_path, args.version, args.epochs, args.learning_rate)
    end_time = time.time() 

    execution_time = end_time - start_time
    print(f"Время обучения и тестирования: {execution_time} секунд")
