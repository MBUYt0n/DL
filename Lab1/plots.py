# code to generate plots

import torch
import tensorflow
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

(x_train, y_train), (x_test, y_test) = tensorflow.keras.datasets.mnist.load_data()
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.uint8)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.uint8)

trainloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train), batch_size=20, shuffle=True
)
validloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_test, y_test), batch_size=20, shuffle=True
)


# for variable model sizes
def model_maker(layers, neurons):
    model_layers = []
    for i in range(layers):
        if i == 0:
            model_layers.append(nn.Linear(784, neurons))
        else:
            model_layers.append(nn.Linear(neurons, neurons))
        model_layers.append(nn.Sigmoid())
    model_layers.append(nn.Linear(neurons, 10))
    model = nn.Sequential(*model_layers)
    return model


# 2 layers 100 neurons
def model_maker1():
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
    ).to("cuda")
    return model


def train_loop(
    model, optimizer, loss, run_name, trainloader=trainloader, validloader=validloader
):
    writer = SummaryWriter(log_dir=f"runs/{run_name}")
    for i in range(50):
        t = 0
        model.train()
        train_progress = tqdm(
            enumerate(trainloader), desc=f"Epoch {i} - Training", total=len(trainloader)
        )
        for j, k in train_progress:
            image, label = k
            image, label = image.to("cuda"), label.to("cuda")
            optimizer.zero_grad()
            output = model(image.view(-1, 784))
            l = loss(output, label)
            l.backward()
            optimizer.step()
            t += l.item()
            train_progress.set_postfix(loss=t / (j + 1))
        t /= len(trainloader)
        writer.add_scalar("Loss/Train", t, i)
        t = 0
        model.eval()
        val_progress = tqdm(
            enumerate(validloader),
            desc=f"Epoch {i} - Validation",
            total=len(validloader),
        )
        for j, k in val_progress:
            image, label = k
            image, label = image.to("cuda"), label.to("cuda")
            output = model(image.view(-1, 784))
            l = loss(output, label)
            t += l.item()
            val_progress.set_postfix(loss=t / (j + 1))
        t /= len(validloader)
        writer.add_scalar("Loss/Validation", t, i)
        writer.flush()
    writer.close()


# for mse
def one_hot_encode(labels, num_classes=10):
    """Convert integer labels to one-hot encoded vectors."""
    labels = labels.type(torch.long)
    return F.one_hot(labels, num_classes=num_classes).float()


# 1,2,3
def layers_and_neurons():
    for i in [1, 2, 3]:
        for j in [50, 100, 200]:
            print(f"Training model with {i} layers and {j} neurons")
            model = model_maker(i, j).to("cuda")
            loss = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            train_loop(model, optimizer, loss, f"layers_{i}_neurons_{j}")


# 4
def optimizers():
    names = ["adam", "nag", "momentum", "sgd"]
    for i in names:
        model = model_maker1()
        if i == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        elif i == "nag":
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.001, nesterov=True, momentum=0.9, dampening=0
            )
        elif i == "momentum":
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
        train_loop(model, optimizer, nn.CrossEntropyLoss(), i)


# 5
def sigmoid_tanh():
    model0 = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
    ).to("cuda")

    model1 = nn.Sequential(
        nn.Linear(784, 100),
        nn.Tanh(),
        nn.Linear(100, 100),
        nn.Tanh(),
        nn.Linear(100, 10),
    ).to("cuda")

    names = ["Sigmoid", "Tanh"]
    models = [model0, model1]
    for name, model in zip(names, models):
        train_loop(
            model,
            torch.optim.Adam(model.parameters(), lr=0.003),
            torch.nn.CrossEntropyLoss(),
            name,
        )


# 6
def cross_entropy():
    losses = [torch.nn.CrossEntropyLoss()]
    for loss in losses:
        model = model_maker1()
        train_loop(
            model,
            torch.optim.Adam(model.parameters(), lr=0.003),
            loss,
            loss.__class__.__name__,
        )


def mse():
    model = nn.Sequential(
        nn.Linear(784, 100),
        nn.Sigmoid(),
        nn.Linear(100, 100),
        nn.Sigmoid(),
        nn.Linear(100, 10),
    ).to("cuda")

    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    writer = SummaryWriter(log_dir=f"runs/MSELoss1")
    for i in range(50):
        train_loss = 0
        model.train()
        train_progress = tqdm(
            enumerate(trainloader), desc=f"Epoch {i} - Training", total=len(trainloader)
        )
        for j, (images, labels) in train_progress:
            images, labels = images.to("cuda"), labels.to("cuda")
            optimizer.zero_grad()
            outputs = nn.functional.softmax(model(images.view(-1, 784)), dim=1)
            labels_one_hot = one_hot_encode(labels, num_classes=10)

            l = loss(outputs, labels_one_hot)
            l.backward()
            optimizer.step()
            train_loss += l.item()
            train_progress.set_postfix(loss=train_loss / (j + 1))
        train_loss /= len(trainloader)
        writer.add_scalar("Loss/Train", train_loss, i)
        val_loss = 0
        model.eval()
        val_progress = tqdm(
            enumerate(validloader),
            desc=f"Epoch {i} - Validation",
            total=len(validloader),
        )
        with torch.no_grad():
            for j, (images, labels) in val_progress:
                images, labels = images.to("cuda"), labels.to("cuda")
                labels_one_hot = one_hot_encode(labels, num_classes=10)
                outputs = nn.functional.softmax(model(images.view(-1, 784)), dim=1)
                l = loss(outputs, labels_one_hot)
                val_loss += l.item()
                val_progress.set_postfix(loss=val_loss / (j + 1))
        val_loss /= len(validloader)
        writer.add_scalar("Loss/Validation", val_loss, i)

        writer.flush()
    writer.close()


# 7
def batch_sizes():
    sizes = [1000, 100, 20, 10]
    for i in sizes:
        trainloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_train, y_train), batch_size=i, shuffle=True
        )
        validloader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(x_test, y_test), batch_size=i, shuffle=True
        )
        model = model_maker1()
        train_loop(
            model,
            torch.optim.Adam(model.parameters(), lr=0.003),
            torch.nn.CrossEntropyLoss(),
            f"{i}",
            trainloader,
            validloader,
        )
