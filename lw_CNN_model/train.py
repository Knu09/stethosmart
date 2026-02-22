import os

from torch import full

# import random
# import torch
# from torch.utils.data import DataLoader
# from dataset import LungDataset
# from lw_cnn_model import LungSoundCNN
# import torch.nn as nn

# -----------------------
# CONFIG
# -----------------------
DATASET_PATH = "../../dataset/ICBHI_final_database/"
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
# Parse Official Split
# -----------------------


def load_icbhi_splits(split_file: str):
    train_subjs = []
    test_subjs = []

    with open(split_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subj, flag = parts
                if flag == "train":
                    train_subjs.append(subj)
                else:
                    test_subjs.append(subj)

    return train_subjs, test_subjs


train_subjs, test_subjs = load_icbhi_splits(
    "../distributions/ICBHI_challenge_train_test_split.txt"
)

# Load Diagnosis Labels


def load_icbhi_labels(label_file: str):
    diag_map = {}
    with open(label_file) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                subj, diag = parts
                diag_map[subj] = diag.lower()

    return diag_map


# Build File List using those Splits
def build_file_list(root, train_subjs, test_subjs, diag_map):
    train_files = []
    test_files = []

    for file in os.listdir(root):
        if file.endswith(".wav"):
            recording_id = file.replace(".wav", "")
            patient_id = recording_id.split("_")[0]

            label = diag_map.get(patient_id, None)
            if label is None:
                continue

            full_path = os.path.join(root, file)

            if recording_id in train_subjs:
                train_files.append((full_path, label))
            elif recording_id in test_subjs:
                test_files.append((full_path, label))

    return train_files, test_files


train_subjs, test_subjs = load_icbhi_splits(
    "../distributions/ICBHI_challenge_train_test_split.txt"
)
diag_map = load_icbhi_labels("../distributions/ICBHI_challenge_diagnosis.txt")
train_files, test_files = build_file_list(
    "../../dataset/ICBHI_final_database/", train_subjs, test_subjs, diag_map
)

# for file in test_files:
#     print(file)


# -----------------------
# Dataset + Loader
# -----------------------
# train_dataset = LungDataset(DATASET_PATH, train_files, label_map)
# test_dataset = LungDataset(DATASET_PATH, test_files, label_map)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)


# # -----------------------
# # Model
# # -----------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# model = LungSoundCNN(num_classes=4).to(device)
#
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
#
#
# # -----------------------
# # TRAIN LOOP (100 EPOCHS)
# # -----------------------
# for epoch in range(EPOCHS):
#
#     model.train()
#     running_loss = 0
#     correct = 0
#     total = 0
#
#     for X, y in train_loader:
#         X, y = X.to(device), y.to(device)
#
#         optimizer.zero_grad()
#         outputs = model(X)
#
#         loss = criterion(outputs, y)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#
#         _, predicted = torch.max(outputs, 1)
#         total += y.size(0)
#         correct += (predicted == y).sum().item()
#
#     train_acc = 100 * correct / total
#
#     # -----------------------
#     # TEST EVALUATION
#     # -----------------------
#     model.eval()
#     test_correct = 0
#     test_total = 0
#
#     with torch.no_grad():
#         for X, y in test_loader:
#             X, y = X.to(device), y.to(device)
#             outputs = model(X)
#
#             _, predicted = torch.max(outputs, 1)
#             test_total += y.size(0)
#             test_correct += (predicted == y).sum().item()
#
#     test_acc = 100 * test_correct / test_total
#
#     print(
#         f"Epoch [{epoch+1}/{EPOCHS}] "
#         f"Loss: {running_loss:.4f} "
#         f"Train Acc: {train_acc:.2f}% "
#         f"Test Acc: {test_acc:.2f}%"
#     )
#
#
# # Save model
# torch.save(model.state_dict(), "lung_model.pth")
# print("Model saved!")
