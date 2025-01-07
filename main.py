import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import typer

from dataset import dataset
from model import my_model

app = typer.Typer()


def plot_accuracy_and_loss(train_loss, train_acc):
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(train_loss)
    axs[0].set_title("Train loss")
    axs[1].plot(train_acc)
    axs[1].set_title("Train accuracy")
    fig.savefig("Training loss and accuracy.png")

@app.command()
def train(lr: float=1e-3):

    batch_size = 32
    epochs = 3

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    train_set, test_set = dataset()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    model = my_model().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()

    print("Training")
    train_loss = []
    train_accuracy = []
    for epoch in range(epochs):
        
        for images, labels in tqdm(iter(train_dataloader)):
        
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, labels)
            train_loss.append(loss.item())

            predictions = output.argmax(dim=1)
            train_accuracy.append((predictions == labels).float().mean().item())

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), "model.pth")

    plot_accuracy_and_loss(train_loss, train_accuracy)

@app.command()
def evaluate(model_checkpoint: str):
    print("Evaluating model")

    batch_size = 32

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    model = my_model().to(device)
    model.load_state_dict(torch.load(model_checkpoint, weights_only=False))

    train_set, test_set = dataset()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(iter(test_dataloader)):

        images, labels = images.to(device), labels.to(device)
        
        output = model(images)

        predictions = output.argmax(dim=1)
        correct += (predictions == labels).float().sum().item()
        total += labels.size(0)
    
    test_accuracy = correct / total
    print(f"Test accuracy: {test_accuracy}")

if __name__ == '__main__':
    
    # train()
    # evaluate("model.pth")
    app()
    print("breddabredda")
    