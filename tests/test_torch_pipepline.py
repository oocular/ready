"""
https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
"""

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

print("data downloads done")

BATCH_SIZE = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

print("data loaders created")

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    """
    Neural Network
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x_):
        """
        forward method for NN
        """
        x_ = self.flatten(x_)
        logits = self.linear_relu_stack(x_)
        return logits


model = NeuralNetwork().to(device)
print(model)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

print("loss function and otimzer defined")


def train(dataloader_, model_, loss_fn_, optimizer_):
    """
    Train method
    """
    size = len(dataloader_.dataset)
    model_.train()
    for batch, (x_, y_) in enumerate(dataloader_):
        x_, y_ = x_.to(device), y_.to(device)

        # Compute prediction error
        pred_ = model_(x_)
        loss = loss_fn_(pred_, y_)

        # Backpropagation
        loss.backward()
        optimizer_.step()
        optimizer_.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(x_)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model_, loss_fn_):
    """
    test method
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model_.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for x_, y_ in dataloader:
            x_, y_ = x_.to(device), y_.to(device)
            pred_ = model_(x_)
            test_loss += loss_fn_(pred_, y_).item()
            correct += (pred_.argmax(1) == y_).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


print("train and test functions defined")

print("training and testing begins ...")

EPOCHS = 3
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done Training!")


# Saving Models
torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")


# Loading Models
model = NeuralNetwork().to(device)
model.load_state_dict(torch.load("model.pth"))

classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    x = x.to(device)
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: "{predicted}", Actual: "{actual}"')
