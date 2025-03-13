import torch
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from src import GraphDataLoader
import time

def load_and_validate_model(model_path, dataset_loader, criterion, config):
    """
    Load a saved model and validate it on a new dataset with additional evaluation metrics.

    Args:
        model_path (str): Path to the saved model (.pth file).
        dataset_loader (DataLoader): The DataLoader for the new dataset.
        criterion (nn.Module): The loss function.
        config (Config): Configuration object containing device settings.
        time_deltas (list): Time intervals for the model input.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    # Initialize the model
    model = torch.load(model_path, map_location=config.device)
    model.to(config.device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in dataset_loader:
            graphs, ips, labels,time_deltas = batch
            graphs = [graph.to(config.device) for graph in graphs]
            ips = [
                {key: val.to(config.device) if isinstance(val, torch.Tensor) else val for key, val in ip_dict.items()}
                for ip_dict in ips
            ]

            # Forward pass
            outputs, global_ips = model(time_deltas, graphs, ips)

            # Prepare labels
            labels_list = []
            for ip in global_ips:
                label = labels[ip]
                labels_list.append(label)
            labels_tensor = torch.tensor(labels_list).to(config.device)

            # Compute loss
            loss = criterion(outputs, labels_tensor)
            total_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_tensor).sum().item()
            total += labels_tensor.size(0)

            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels_tensor.cpu().numpy())
    print(len(all_labels))
    # Compute final metrics
    avg_loss = total_loss / len(dataset_loader)
    accuracy = correct / total

    # Precision, Recall, F1-score
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return {
        "Average Loss": avg_loss,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Confusion Matrix": conf_matrix
    }

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src import CombinedModel, GraphDataLoader, Config
    import torch.nn as nn

    # Configuration
    config = Config()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights).to(config.device))

    # Load dataset
    data_loader = GraphDataLoader(config.dataPath, None, None)
    data_loader.process_data()
    new_dataset_loader = data_loader.get_valid_loader()  # Replace with appropriate dataset

    # Validate/home/lzy/Gnn/DHGNN-LSTM/result/20250312-154508/training_log.txt
    model_path = "../result/20250313-003952/model.pth"
    metrics = load_and_validate_model(
        model_path,
        new_dataset_loader,
        criterion,
        config
    )

    # Print results
    print(f"Average Loss: {metrics['Average Loss']:.4f}")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Precision: {metrics['Precision']:.4f}")
    print(f"Recall: {metrics['Recall']:.4f}")
    print(f"F1 Score: {metrics['F1 Score']:.4f}")
    print("Confusion Matrix:\n", metrics["Confusion Matrix"])
