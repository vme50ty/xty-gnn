'''
Author: lee12345 15116908166@163.com
Date: 2025-01-02 10:50:28
LastEditors: lee12345 15116908166@163.com
LastEditTime: 2025-01-02 14:34:33
FilePath: /Gnn/DHGNN-LSTM/Codes/model_evaluate.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
from src import GraphDataLoader

def load_and_validate_model(model_path, dataset_loader, criterion, config, time_deltas):
    """
    Load a saved model and validate it on a new dataset.

    Args:
        model_path (str): Path to the saved model (.pth file).
        model_class (class): The class of the model to initialize.
        dataset_loader (DataLoader): The DataLoader for the new dataset.
        criterion (nn.Module): The loss function.
        device (torch.device): The device to run the validation on (CPU or GPU).
        time_deltas (list): Time intervals for the model input.

    Returns:
        float: Average loss over the dataset.
        float: Accuracy of the model on the dataset.
    """
    # Initialize the model
    model = torch.load(model_path, map_location=config.device)
    model.to(config.device)
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataset_loader:
            graphs, ips, labels = batch
            graphs = [graph.to(device) for graph in graphs]
            ips = [
                {key: val.to(device) if isinstance(val, torch.Tensor) else val for key, val in ip_dict.items()}
                for ip_dict in ips
            ]

            # Forward pass
            outputs, global_ips = model(time_deltas, graphs, ips)

            # Prepare labels
            labels_list = []
            for ip in global_ips:
                label = labels[ip]
                labels_list.append(label)
            labels_tensor = torch.tensor(labels_list).to(device)

            loss = criterion(outputs, labels_tensor)

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels_tensor).sum().item()
            total += labels_tensor.size(0)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataset_loader)
    accuracy = correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from src import CombinedModel, GraphDataLoader, Config
    import torch.nn as nn

    # Configuration
    config = Config()
    device = config.device
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(config.class_weights).to(device))

    # Load dataset
    data_loader = GraphDataLoader(config.dataPath, None, None)
    data_loader.process_data()
    new_dataset_loader = data_loader.get_valid_loader()  # Replace with appropriate dataset

    # Validate
    model_path = "../result/20250102-112438/model.pth"
    avg_loss, accuracy = load_and_validate_model(
        model_path,
        new_dataset_loader,
        criterion,
        config,
        time_deltas=[5, 10, 10, 10, 10, 10, 10, 10, 10]
    )

    print(f"Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")