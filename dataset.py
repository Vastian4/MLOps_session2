import torch

data_path = './data/corruptedmnist'

def dataset():

    train_data = []
    train_labels = []

    for i in range(6):
        train_data.append(torch.load(f"{data_path}/train_images_{i}.pt", weights_only=False))
        train_labels.append(torch.load(f"{data_path}/train_target_{i}.pt", weights_only=False))
    
    train_data = torch.cat(train_data)
    train_labels = torch.cat(train_labels)

    test_data = torch.load(f"{data_path}/test_images.pt", weights_only=False)
    test_labels = torch.load(f"{data_path}/test_target.pt", weights_only=False)

    train_data = train_data.unsqueeze(1).float()
    test_data = test_data.unsqueeze(1).float()

    train_data = torch.utils.data.TensorDataset(train_data, train_labels)
    test_data = torch.utils.data.TensorDataset(test_data, test_labels)

    return train_data, test_data


