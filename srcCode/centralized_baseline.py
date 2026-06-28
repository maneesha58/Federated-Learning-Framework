"""
Centralized Baseline Experiment for Federated Learning
======================================================
Trains a simple 2-layer NN on FashionMNIST (centralized, no federation).
Runs 5 independent trials and reports mean/variance of final test accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend so script completes in terminal
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 10
INPUT_SIZE = 28 * 28  # 784
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
NUM_RUNS = 5
RANDOM_SEED = 42


def get_dataloaders(batch_size=64):
    """
    Create FashionMNIST train and test DataLoaders with standard normalization.
    """
    # Standard normalization: (x - mean) / std; common for MNIST-style [0,1] -> ~[-1,1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_model(device):
    """
    Build a fresh 2-layer neural network:
    Input (784) -> Hidden (128) + ReLU -> Output (10).
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    )
    return model.to(device)


def train(model, device, train_loader, optimizer, criterion):
    """
    Run one epoch of training. Returns average training accuracy over the epoch.
    """
    model.train()
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def test(model, device, test_loader, criterion):
    """
    Evaluate model on test set. Returns test accuracy (no gradient).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    return accuracy


def run_experiment(device, train_loader, test_loader, run_id=0):
    """
    Run one full experiment: fresh model, 20 epochs.
    Returns (train_accs, test_accs) per epoch and final_test_acc.
    """
    model = get_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_accs = []
    test_accs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_acc = train(model, device, train_loader, optimizer, criterion)
        test_acc = test(model, device, test_loader, criterion)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(f"  Run {run_id + 1} | Epoch {epoch:2d}/{NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%", flush=True)

    final_test_acc = test_accs[-1]
    return train_accs, test_accs, final_test_acc


def plot_run(run_id, train_accs, test_accs, save_path=None):
    """
    Plot training and test accuracy vs epoch for one run.
    """
    epochs = np.arange(1, len(train_accs) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_accs, "b-o", markersize=4, label="Training accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Run {run_id + 1}: Training Accuracy vs Epoch")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, test_accs, "g-s", markersize=4, label="Test accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Run {run_id + 1}: Test Accuracy vs Epoch")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}", flush=True)
    plt.close()


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    train_loader, test_loader = get_dataloaders(batch_size=BATCH_SIZE)
    print(f"Training batches: {len(train_loader)}, Test batches: {len(test_loader)}\n")

    all_train_accs = []
    all_test_accs = []
    final_test_accs = []

    for run in range(NUM_RUNS):
        print(f"--- Run {run + 1}/{NUM_RUNS} ---")
        train_accs, test_accs, final_test = run_experiment(
            device, train_loader, test_loader, run_id=run
        )
        all_train_accs.append(train_accs)
        all_test_accs.append(test_accs)
        final_test_accs.append(final_test)
        plot_run(run, train_accs, test_accs, save_path=f"run_{run + 1}_accuracy.png")
        print()

    # Aggregate results
    final_test_accs = np.array(final_test_accs)
    mean_final = np.mean(final_test_accs)
    var_final = np.var(final_test_accs)

    print("=" * 60)
    print("CENTRALIZED BASELINE — SUMMARY (5 RUNS)")
    print("=" * 60)
    print(f"Final test accuracy per run: {final_test_accs}")
    print(f"Average final test accuracy: {mean_final:.2f}%")
    print(f"Variance of final test accuracy: {var_final:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
