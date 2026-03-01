import torch
import torch.nn as nn
from torch import full
from torch.utils.data import TensorDataset, DataLoader
from lw_cnn_model import LungSoundCNN

from collections import Counter
import numpy as np
from preprocessing import (
    segment_cycles,
    load_icbhi_splits,
    load_icbhi_labels,
    build_file_list,
    build_cycle_dataset,
    extract_features,
)
from typing import List, Dict
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import json

# import random
# import torch
# from torch.utils.data import DataLoader
# from dataset import LungDataset
# from lw_cnn_model import LungSoundCNN
# import torch.nn as nn

# -----------------------
# CONFIG
# -----------------------
BATCH_SIZE = 16
EPOCHS = 100
LR = 1e-4

# -----------------------
# PATHS
# -----------------------
ICBHI_SPLITS_PATH = "../distributions/ICBHI_challenge_train_test_split.txt"
ICBHI_DIAGNOSIS_PATH = "../distributions/ICBHI_challenge_diagnosis.txt"
ICBHI_DATASET_PATH = "../../dataset/ICBHI_final_database/"


def main():
    train_subjs, test_subjs = load_icbhi_splits(ICBHI_SPLITS_PATH)

    train_subjs, test_subjs = load_icbhi_splits(ICBHI_SPLITS_PATH)
    diag_map = load_icbhi_labels(ICBHI_DIAGNOSIS_PATH)
    train_files, test_files = build_file_list(
        ICBHI_DATASET_PATH, train_subjs, test_subjs, diag_map
    )

    # Build cycle-level dataset
    train_audio, train_labels = build_cycle_dataset(train_files)
    test_audio, test_labels = build_cycle_dataset(test_files)

    # Extract train and test features
    print("Feature extracting the train and test independent features...")

    train_features = []
    for i, c in enumerate(train_audio):
        print(f"[TRAIN {i + 1}/{len(train_audio)}] Extracting: {c}")
        train_features.append(extract_features(c))

    test_features = []
    for i, c in enumerate(test_audio):
        print(f"[TEST {i + 1}/{len(test_audio)}] Extracting: {c}")
        test_features.append(extract_features(c))

    print("Feature extraction completed.\n")

    # Convert to tensors
    X_train = torch.tensor(np.array(train_features)).float()
    y_train = torch.tensor(train_labels).long()

    X_test = torch.tensor(np.array(test_features)).float()
    y_test = torch.tensor(test_labels).long()

    # Normalization
    # This improves training stability significantly.
    mean = X_train.mean()
    std = X_train.std()

    X_train = (X_train - mean) / (std + 1e-8)
    X_test = (X_test - mean) / (std + 1e-8)

    # -----------------------
    # Dataset + Loader
    # -----------------------
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # -----------------------
    # Model
    # -----------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LungSoundCNN(num_classes=4).to(device)

    # Class imbalance
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())

    class_counts = np.array(
        [
            train_labels.count(i) if train_labels.count(i) > 0 else 1  # avoid zero
            for i in range(4)
        ]
    )
    total_samples = sum(class_counts)
    print(f"Total samples: {total_samples}")

    class_weights = torch.tensor(
        [total_samples / c for c in class_counts], dtype=torch.float
    )

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # -----------------------
    # TRAIN LOOP (100 EPOCHS)
    # -----------------------
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    print("Starting the training loop...")
    for epoch in range(EPOCHS):
        print(f"Training epoch [{epoch + 1}]")
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

        train_losses.append(running_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Loss: {running_loss:.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Test Acc: {test_acc:.2f}%"
        )

    # Save model
    torch.save(model.state_dict(), "lung_model.pth")
    print("Model saved!")
    epochs = range(1, EPOCHS + 1)

    plt.figure(figsize=(14, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.close()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.tight_layout()
    plt.savefig("loss_plot.png", dpi=300)
    plt.close()

    metrics = {
        "train_loss": train_losses,
        "train_acc": train_accuracies,
        "test_acc": test_accuracies,
    }

    with open("training_metrics.json", "w") as f:
        json.dump(metrics, f)

    # total_cycles = 0
    # for file, label in train_files:
    #     print(f"File Path: {file}")
    #     print(f"Label: {label}")
    #     cycles = segment_cycles(file)
    #     print(f"Length of cycles: {len(cycles)}")
    #
    #     if cycles is not None:
    #         total_cycles += len(cycles)
    #
    # print(f"Total lenght of cycles: {total_cycles}")


if __name__ == "__main__":
    main()
    main()
