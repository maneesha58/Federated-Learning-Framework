"""
Federated Learning with FedAvg on FashionMNIST (Question 2)
===========================================================
Simulates 5 clients with non-IID data; server aggregates with weighted FedAvg.
20 communication rounds per run; 5 runs total. Same 2-layer NN as centralized baseline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


# ---------------------------------------------------------------------------
# Configuration (match centralized baseline where applicable)
# ---------------------------------------------------------------------------
BATCH_SIZE = 64
HIDDEN_SIZE = 128
NUM_CLASSES = 10
INPUT_SIZE = 28 * 28
LEARNING_RATE = 0.001
NUM_CLIENTS = 5
NUM_ROUNDS = 20
NUM_RUNS = 5
RANDOM_SEED = 42
# Non-IID: fraction of each label's samples that go to that label's "primary" client
PRIMARY_FRACTION = 0.85


def get_model(device):
    """
    Same 2-layer NN as centralized baseline:
    Input (784) -> Hidden (128) + ReLU -> Output (10).
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
        nn.ReLU(),
        nn.Linear(HIDDEN_SIZE, NUM_CLASSES),
    )
    return model.to(device)


def create_non_iid_client_datasets(train_dataset, num_clients=5, primary_fraction=0.85, seed=42):
    """
    Split training data by label unevenly across clients (non-IID).
    Each client gets a majority of 2 labels (primary) and fewer of the rest.
    No overlapping samples between clients.
    """
    rng = np.random.default_rng(seed)
    # Group indices by label (FashionMNIST labels 0..9)
    indices_by_label = [[] for _ in range(NUM_CLASSES)]
    for idx in range(len(train_dataset)):
        _, y = train_dataset[idx]
        indices_by_label[y].append(idx)
    indices_by_label = [np.array(arr) for arr in indices_by_label]

    # Assign each label unevenly: primary client gets primary_fraction, rest split among others
    client_indices = [[] for _ in range(num_clients)]
    for label in range(NUM_CLASSES):
        primary_client = label // 2  # 0,1->0; 2,3->1; 4,5->2; 6,7->3; 8,9->4
        inds = indices_by_label[label].copy()
        rng.shuffle(inds)
        n = len(inds)
        n_primary = max(1, int(n * primary_fraction))
        # Primary client gets n_primary samples; no overlap
        client_indices[primary_client].extend(inds[:n_primary].tolist())
        # Remaining samples go to the other 4 clients in round-robin (no overlap)
        rest = inds[n_primary:]
        others = [c for c in range(num_clients) if c != primary_client]
        for i, idx in enumerate(rest):
            c = others[i % len(others)]
            client_indices[c].append(int(idx))

    # Shuffle each client's indices
    for k in range(num_clients):
        arr = np.array(client_indices[k])
        rng.shuffle(arr)
        client_indices[k] = arr.tolist()

    return client_indices


def get_client_loaders_and_test(train_dataset, client_indices, batch_size=64, test_dataset=None):
    """
    Build one DataLoader per client (Subset of train_dataset) and optional test_loader.
    """
    client_loaders = []
    for inds in client_indices:
        subset = Subset(train_dataset, inds)
        client_loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=0))
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return client_loaders, test_loader


# ---------------------------------------------------------------------------
# Client: holds local data and trains one epoch with current global model
# ---------------------------------------------------------------------------
class Client:
    """
    Simulates one federated client: owns local data and can perform local training.
    """

    def __init__(self, client_id, train_loader, device):
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.n_local = sum(len(batch[1]) for batch in train_loader)

    def train_one_epoch(self, model, criterion, lr=0.001):
        """
        FedAvg step (b): Client trains locally for 1 epoch.
        Receives (a copy of) global model, trains on local data, returns updated state_dict.
        """
        local_model = deepcopy(model)
        local_model.train()
        optimizer = optim.Adam(local_model.parameters(), lr=lr)
        correct = 0
        total = 0
        for data, target in self.train_loader:
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = local_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        train_acc = 100.0 * correct / total if total > 0 else 0.0
        # FedAvg step (c): Return updated weights (and local count for aggregation)
        return local_model.state_dict(), train_acc, total


def aggregate_weights(client_weights_list, client_counts):
    """
    FedAvg step (d): Server aggregates client weights using weighted average.
    global_weight = sum(n_k / n_total * client_weight_k)
    """
    n_total = sum(client_counts)
    if n_total == 0:
        raise ValueError("Total sample count is zero")
    # Use first client's state_dict as template for key names
    aggregated = {}
    for key in client_weights_list[0].keys():
        aggregated[key] = sum(
            (n_k / n_total) * client_weights_list[i][key].float()
            for i, n_k in enumerate(client_counts)
        )
    return aggregated


def evaluate(model, device, test_loader):
    """
    Evaluate global model on the test set. Returns test accuracy (%).
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


def run_federated(device, client_loaders, test_loader, num_rounds=20, lr=0.001, run_id=0):
    """
    Run one full FedAvg experiment: num_rounds communication rounds.
    Returns:
      test_accs: list of test accuracy after each round
      client_train_accs: list of lists; client_train_accs[r][k] = client k train acc at round r
    """
    criterion = nn.CrossEntropyLoss()
    # Initialize global model (server)
    global_model = get_model(device)
    clients = [Client(k, loader, device) for k, loader in enumerate(client_loaders)]

    test_accs = []
    client_train_accs = []  # per round: list of 5 accs

    for round_ in range(num_rounds):
        # FedAvg step (a): Server sends global model to all clients (we pass a copy in train_one_epoch)
        client_weights_list = []
        client_counts = []
        round_train_accs = []

        for client in clients:
            w, train_acc, n = client.train_one_epoch(global_model, criterion, lr=lr)
            client_weights_list.append(w)
            client_counts.append(n)
            round_train_accs.append(train_acc)

        client_train_accs.append(round_train_accs)

        # FedAvg step (d): Server aggregates weights
        aggregated_state = aggregate_weights(client_weights_list, client_counts)
        global_model.load_state_dict(aggregated_state)

        # After each round: evaluate global model on test set
        test_acc = evaluate(global_model, device, test_loader)
        test_accs.append(test_acc)
        print(
            f"  Run {run_id + 1} | Round {round_ + 1:2d}/{num_rounds} | Test: {test_acc:.2f}% | "
            f"Clients train: [{', '.join(f'{a:.1f}' for a in round_train_accs)}]%",
            flush=True,
        )

    return test_accs, client_train_accs


def plot_run(run_id, test_accs, client_train_accs, num_rounds, save_path=None):
    """
    Plot server test accuracy vs rounds and each client's training accuracy vs rounds.
    """
    rounds = np.arange(1, num_rounds + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # Server test accuracy vs communication rounds
    ax1.plot(rounds, test_accs, "b-o", markersize=4, label="Server test accuracy")
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title(f"Run {run_id + 1}: Server (global) test accuracy vs communication rounds")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Each client's training accuracy vs rounds
    for k in range(len(client_train_accs[0])):
        accs = [client_train_accs[r][k] for r in range(num_rounds)]
        ax2.plot(rounds, accs, "-", markersize=3, label=f"Client {k}")
    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_title(f"Run {run_id + 1}: Each client's local training accuracy vs rounds")
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
    print(f"Device: {device}")
    print("Non-IID split: each client has majority of 2 labels (primary), rest shared.\n")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_dataset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

    all_final_test_accs = []

    for run in range(NUM_RUNS):
        # Fresh non-IID split and loaders each run (different shuffle)
        client_indices = create_non_iid_client_datasets(
            train_dataset, num_clients=NUM_CLIENTS, primary_fraction=PRIMARY_FRACTION, seed=RANDOM_SEED + run
        )
        client_loaders, test_loader = get_client_loaders_and_test(
            train_dataset, client_indices, batch_size=BATCH_SIZE, test_dataset=test_dataset
        )
        print(f"--- Run {run + 1}/{NUM_RUNS} ---")
        test_accs, client_train_accs = run_federated(
            device, client_loaders, test_loader, num_rounds=NUM_ROUNDS, lr=LEARNING_RATE, run_id=run
        )
        all_final_test_accs.append(test_accs[-1])
        plot_run(
            run,
            test_accs,
            client_train_accs,
            NUM_ROUNDS,
            save_path=f"fedavg_run_{run + 1}.png",
        )
        print()

    # Summary over 5 runs
    all_final_test_accs = np.array(all_final_test_accs)
    mean_final = np.mean(all_final_test_accs)
    var_final = np.var(all_final_test_accs)

    print("=" * 60)
    print("FEDAVG FASHIONMNIST — SUMMARY (5 RUNS)")
    print("=" * 60)
    print(f"Final test accuracy per run: {all_final_test_accs}")
    print(f"Average final test accuracy: {mean_final:.2f}%")
    print(f"Variance of final test accuracy: {var_final:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
