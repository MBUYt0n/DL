# Code to generate submission.csv file

import torch
import tensorflow
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

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

model = nn.Sequential(
    nn.Linear(784, 100),
    nn.Sigmoid(),
    nn.Linear(100, 100),
    nn.Sigmoid(),
    nn.Linear(100, 100),
    nn.Sigmoid(),
    nn.Linear(100, 10),
).to("cuda")

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=5, min_lr=0.0005
)


previous_lr = optimizer.param_groups[0]["lr"]

for i in range(100):
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

    current_lr = optimizer.param_groups[0]["lr"]
    if current_lr != previous_lr:
        print(f"Learning Rate changed to {current_lr}")
        previous_lr = current_lr

    t = 0
    model.eval()
    val_progress = tqdm(
        enumerate(validloader), desc=f"Epoch {i} - Validation", total=len(validloader)
    )
    for j, k in val_progress:
        image, label = k
        image, label = image.to("cuda"), label.to("cuda")
        output = model(image.view(-1, 784))
        l = loss(output, label)
        t += l.item()
        val_progress.set_postfix(loss=t / (j + 1))
    t /= len(validloader)
    lr_scheduler.step(t)

# Test accuracy
t = 0
for j, k in zip(x_test, y_test):
    image = j.to("cuda")
    with torch.no_grad():
        output = model(image.view(-1, 784))
    output = torch.nn.Softmax(dim=1)(output)
    output = torch.argmax(output).to("cpu")
    if output == k:
        t += 1


print(t / len(y_test))

# cconfusion matrix
num_classes = 10
all_preds = []
all_labels = []

for images, labels in validloader:
    images, labels = images.to("cuda"), labels.to("cuda")

    with torch.no_grad():
        outputs = model(images.view(-1, 784))
        outputs = torch.nn.Softmax(dim=1)(outputs)
        preds = torch.argmax(outputs, dim=1)

    all_preds.extend(preds.cpu().numpy())
    all_labels.extend(labels.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds, labels=np.arange(num_classes))

plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=np.arange(num_classes),
    yticklabels=np.arange(num_classes),
)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()


# Submission
outputs = []
for j, k in zip(x_test, y_test):
    image = j.to("cuda")
    with torch.no_grad():
        output = model(image.view(-1, 784))
    output = torch.nn.Softmax(dim=1)(output)
    output = torch.argmax(output).to("cpu")
    outputs.append(int(output))

with open("test_submission.csv", "w") as f:
    f.write("id,label\n")
    for i, j in enumerate(outputs):
        f.write(f"{i},{j}\n")
