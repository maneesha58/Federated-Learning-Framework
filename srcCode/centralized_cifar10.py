"""
Centralized Baseline Experiment for CIFAR-10
=============================================
Trains a small CNN on CIFAR-10 (centralized). Runs 5 independent trials
and reports mean/variance of final test accuracy. Optimized for CPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
NUM_EPOCHS = 20
NUM_RUNS = 5
RANDOM_SEED = 42
# CIFAR-10 standard normalization (mean, std per channel)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


class CNNModel(nn.Module):
    """
    Small CNN for CIFAR-10 (32x32 RGB):
    - Conv 3->32, ReLU, MaxPool -> 16x16
    - Conv 32->64, ReLU, MaxPool -> 8x8
    - Flatten -> 64*8*8 = 4096
    - Linear 4096 -> 128, ReLU
    - Linear 128 -> 10
    """

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32 -> 16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_dataloaders(batch_size=64):
    """
    Create CIFAR-10 train and test DataLoaders with standard normalization.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    test_set = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader


def train(model, device, train_loader, optimizer, criterion):
    """
    Run one epoch of training. Returns training accuracy (%).
    """
    model.train()
    correct = 0
    total = 0
    for data, target in train_loader:
        data, target = data.to(device, non_blocking=False), target.to(device, non_blocking=False)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
    return 100.0 * correct / total if total > 0 else 0.0


def test(model, device, test_loader):
    """
    Evaluate model on test set. Returns test accuracy (%).
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
    return 100.0 * correct / total if total > 0 else 0.0


def run_experiment(device, train_loader, test_loader, run_id=0):
    """
    Run one full experiment: fresh CNN, 20 epochs. Returns train_accs, test_accs, final_test_acc.
    """
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_accs = []
    test_accs = []

    for epoch in range(1, NUM_EPOCHS + 1):
        train_acc = train(model, device, train_loader, optimizer, criterion)
        test_acc = test(model, device, test_loader)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        print(
            f"  Run {run_id + 1} | Epoch {epoch:2d}/{NUM_EPOCHS} | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%",
            flush=True,
        )

    return train_accs, test_accs, test_accs[-1]


def plot_run(run_id, train_accs, test_accs, save_path=None):
    """
    Plot training and test accuracy vs epoch for one run.
    """
    epochs = np.arange(1, len(train_accs) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_accs, "b-o", markersize=4, label="Training accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Run {run_id + 1}: Training Accuracy vs Epoch (CIFAR-10)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, test_accs, "g-s", markersize=4, label="Test accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Run {run_id + 1}: Test Accuracy vs Epoch (CIFAR-10)")
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

    final_test_accs = []

    for run in range(NUM_RUNS):
        print(f"--- Run {run + 1}/{NUM_RUNS} ---")
        train_accs, test_accs, final_test = run_experiment(
            device, train_loader, test_loader, run_id=run
        )
        final_test_accs.append(final_test)
        plot_run(run, train_accs, test_accs, save_path=f"centralized_cifar10_run_{run + 1}.png")
        print()

    final_test_accs = np.array(final_test_accs)
    mean_final = np.mean(final_test_accs)
    var_final = np.var(final_test_accs)

    print("=" * 60)
    print("CENTRALIZED CIFAR-10 — SUMMARY (5 RUNS)")
    print("=" * 60)
    print(f"Final test accuracy per run: {final_test_accs}")
    print(f"Average final test accuracy: {mean_final:.2f}%")
    print(f"Variance of final test accuracy: {var_final:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
