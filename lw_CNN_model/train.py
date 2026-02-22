import os
import random
import torch
from torch.utils.data import DataLoader
from dataset import LungDataset
from lw_cnn_model import LungSoundCNN
import torch.nn as nn

# -----------------------
# CONFIG
# -----------------------
DATASET_PATH = "dataset"
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4


# Label encoding
label_map = {
    "normal": 0,
    "copd": 1,
    "asthma": 2,
    "pneumonia": 3,
}


# -----------------------
# Collect file list
# -----------------------
file_list = []

for label in label_map.keys():
    class_dir = os.path.join(DATASET_PATH, label)
    for file in os.listdir(class_dir):
        if file.endswith(".wav"):
            file_list.append((os.path.join(label, file), label))

# Shuffle
random.shuffle(file_list)

# 80:20 split
split_idx = int(0.8 * len(file_list))
train_files = file_list[:split_idx]
test_files = file_list[split_idx:]


# -----------------------
# Dataset + Loader
# -----------------------
train_dataset = LungDataset(DATASET_PATH, train_files, label_map)
test_dataset = LungDataset(DATASET_PATH, test_files, label_map)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# -----------------------
# Model
# -----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = LungSoundCNN(num_classes=4).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# -----------------------
# TRAIN LOOP (100 EPOCHS)
# -----------------------
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()

    train_acc = 100 * correct / total

    # -----------------------
    # TEST EVALUATION
    # -----------------------
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            _, predicted = torch.max(outputs, 1)
            test_total += y.size(0)
            test_correct += (predicted == y).sum().item()

    test_acc = 100 * test_correct / test_total

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {running_loss:.4f} "
        f"Train Acc: {train_acc:.2f}% "
        f"Test Acc: {test_acc:.2f}%"
    )


# Save model
torch.save(model.state_dict(), "lung_model.pth")
print("Model saved!")
