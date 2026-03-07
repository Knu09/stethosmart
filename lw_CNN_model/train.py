import os
import torch
import torch.nn as nn
from torch import full
from torch.utils.data import TensorDataset, DataLoader
from lw_cnn_model import LungSoundCNN
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from collections import Counter
import numpy as np
from preprocessing import (
    collect_segments,
    inject_icbhi_diagnosis,
    segment_cycles,
    load_icbhi_splits,
    load_icbhi_labels,
    build_file_list,
    build_cycle_dataset,
    extract_features,
    collect_segments,
    undersample,
    balance_by_augmentation,
)
from typing import List, Dict
from numpy.typing import NDArray

import matplotlib.pyplot as plt
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

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
KAUH_DATASET_PATH = "../../dataset/KAUH_final_database/Audio Files/"
ICBHI_DATASET_PATH = "../../dataset/ICBHI_final_database/"


def balance_training_data(X_train, y_train):

    print("Original distribution:")
    print_class_distribution(y_train)

    # Step 1: Undersample COPD
    X_under, y_under = undersample(X_train, y_train)

    print("After COPD undersampling:")
    print_class_distribution(y_under)

    # Step 2: Augment minority classes
    X_bal, y_bal = balance_by_augmentation(X_under, y_under)

    print("Final balanced distribution:")
    print_class_distribution(y_bal)

    return X_bal, y_bal


def print_class_distribution(labels):
    counter = Counter(labels)
    for k, v in counter.items():
        print(f"Class {k}: {v}")


def split_data(segments, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        segments, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"Train segments {len(X_train)}")
    print(f"Test segments {len(X_test)}")

    return X_train, X_test, y_train, y_test


def main():
    # NOTE: trigger this method for the first time, then comment it after running python train.py.
    # inject_icbhi_diagnosis(diag_map, ICBHI_DATASET_PATH)

    diag_map = load_icbhi_labels(ICBHI_DIAGNOSIS_PATH)
    all_segments, all_labels = collect_segments(
        ICBHI_DATASET_PATH, diag_map, KAUH_DATASET_PATH
    )

    print("Total segments: ", len(all_segments))

    # Split the dataset
    X_train, X_test, y_train, y_test = split_data(all_segments, all_labels)

    # Balance training data
    # returns balanced segmented samples and labels
    X_train_bal, y_train_bal = balance_training_data(X_train, y_train)
    print("Balanced train size:", len(X_train_bal))
    print("Test size:", len(X_test))

    # train_files, test_files = build_file_list(
    #     ICBHI_DATASET_PATH, train_subjs, test_subjs, diag_map
    # )

    # # Build cycle-level dataset
    # train_audio, train_labels = build_cycle_dataset(train_files)
    # test_audio, test_labels = build_cycle_dataset(test_files)
    #
    # # print("Train distribution:", Counter(train_labels))
    # # print("Test distribution:", Counter(test_labels))
    #
    # Extract train and test features
    print("Feature extracting the train and test independent features...")

    train_features = []
    test_features = []
    # for i, c in enumerate(X_train_bal):
    #     print(f"[TRAIN {i + 1}/{len(X_train_bal)}] Extracting: {c}")
    #     train_features.append(extract_features(c))
    #
    # for i, c in enumerate(X_test):
    #     print(f"[TEST {i + 1}/{len(X_test)}] Extracting: {c}")
    #     test_features.append(extract_features(c))

    for c in tqdm(X_train_bal, desc="Extracting train features"):
        train_features.append(extract_features(c))

    for c in tqdm(X_test, desc="Extracting test features"):
        test_features.append(extract_features(c))

    print("Feature extraction completed.\n")

    train_model(train_features, y_train_bal, test_features, y_test)


def train_model(train_features, train_labels, test_features, test_labels):
    # Label encoder
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.transform(test_labels)

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

    class_counts = np.array([max(np.sum(train_labels == i), 1) for i in range(4)])

    total_samples = sum(class_counts)
    print(f"Total train samples: {total_samples}")

    class_weights = torch.tensor(
        [total_samples / c for c in class_counts], dtype=torch.float, device=device
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
    epoch_bar = tqdm(range(EPOCHS), desc="Training Progress")

    for epoch in epoch_bar:
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

        epoch_bar.set_postfix(
            loss=running_loss,
            train_acc=f"{train_acc:.2f}%",
            test_acc=f"{test_acc:.2f}%",
        )

    # Save scaler
    mean = X_train.mean().item()
    std = X_train.std().item()
    torch.save({"mean": mean, "std": std}, "scaler.pth")

    # Save model
    torch.save(model.state_dict(), "lung_model.pth")
    print("Model saved!")

    epochs = range(1, EPOCHS + 1)

    # -----------------------
    # CONFUSION MATRIX
    # -----------------------
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)

            outputs = model(X)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))

    class_names = ["Healthy", "COPD", "Asthma", "Pneumonia"]

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=300)
    plt.close()

    print("Confusion matrix saved!")

    # -----------------------
    # Accuracy Plot
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accuracies, label="Train Accuracy")
    plt.plot(epochs, test_accuracies, label="Test Accuracy")

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Test Accuracy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("accuracy_plot.png", dpi=300)
    plt.close()

    # -----------------------
    # Loss Plot
    # -----------------------
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Training Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.grid(True)

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
