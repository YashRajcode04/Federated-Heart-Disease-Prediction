# fl_heart.py
print("✅ Script started")

import os
import numpy as np
import pandas as pd
import torch
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from fl_algorithms import HeartModel, state_dict_to_cpu, fedavg_aggregate, fedper_aggregate, get_backbone_state_dict, set_backbone_state_dict

# ---------- Config ----------
DATA_PATH = "C:/Users/yashr/OneDrive/Desktop/Heart_2022/data/heart_2022_no_nans.csv"
N_CLIENTS = 4
R = 20      # rounds
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LR = 1e-3
DEVICE = "cpu"  # change to "cuda" if you have GPU
DIRICHLET_ALPHA = None  # set float (e.g., 0.3) for Dirichlet non-IID split or None for equal split
SEED = 1
PROX_MU = 0.01  # FedProx mu

# ---------- Repro ----------
def seed_everything(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(SEED)

# ---------- Data loading ----------
# ---------- Data loading ----------
def load_heart(path):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(path)

    # Last column is the label
    y = df.iloc[:, -1]

    # Map Yes/No or other string labels to integers
    if y.dtype == object:
        y = y.map({"No": 0, "Yes": 1})
    else:
        y = y.astype(int)

    # Features = all except last column
    X = df.iloc[:, :-1]

    # Convert all categorical columns to numeric
    X = pd.get_dummies(X, drop_first=True)

    # Ensure all columns are numeric
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop rows with any NaN values
    data_df = pd.concat([X, y], axis=1).dropna(axis=0)
    X = data_df.iloc[:, :-1]
    y = data_df.iloc[:, -1]

    # Standardize numeric features
    X_scaled = StandardScaler().fit_transform(X)

    return X_scaled, y.values

# ---------- Load data ----------
X, y = load_heart(DATA_PATH)

# ---------- Train/Test Split ----------
from sklearn.model_selection import train_test_split
SEED = 1
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)


# ---------- Partition into clients ----------
def equal_split_indices(n_samples, n_clients):
    sizes = [n_samples // n_clients] * n_clients
    for i in range(n_samples % n_clients):
        sizes[i] += 1
    idx = np.random.permutation(n_samples)
    pos = 0
    splits = []
    for s in sizes:
        splits.append(idx[pos:pos+s])
        pos += s
    return splits

def dirichlet_split(y, n_clients, alpha=0.3, min_size=10, seed=0):
    np.random.seed(seed)
    y = np.array(y)
    classes = np.unique(y)
    client_indices = [[] for _ in range(n_clients)]
    for c in classes:
        idx_c = np.where(y == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * n_clients)
        counts = (proportions * len(idx_c)).astype(int)
        while counts.sum() < len(idx_c):
            counts[np.argmax(proportions)] += 1
        start = 0
        for i, cnt in enumerate(counts):
            if cnt > 0:
                client_indices[i].extend(idx_c[start:start+cnt].tolist())
                start += cnt
    # ensure min size
    for i in range(n_clients):
        if len(client_indices[i]) < min_size:
            donor = np.argmax([len(c) for c in client_indices])
            need = min_size - len(client_indices[i])
            move = client_indices[donor][:need]
            client_indices[i].extend(move)
            client_indices[donor] = client_indices[donor][need:]
    client_indices = [np.array(sorted(x)) for x in client_indices]
    return client_indices

if DIRICHLET_ALPHA is None:
    client_splits = equal_split_indices(len(X_train), N_CLIENTS)
else:
    client_splits = dirichlet_split(y_train, N_CLIENTS, alpha=DIRICHLET_ALPHA, seed=SEED)

# ---------- Create client datasets ----------
client_loaders = []
client_sizes = []

for idx in client_splits:
    Xi = torch.tensor(X_train[idx], dtype=torch.float32)
    yi = torch.tensor(y_train[idx], dtype=torch.float32)
    ds = TensorDataset(Xi, yi)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    client_loaders.append(loader)
    client_sizes.append(len(idx))

# ---------- Global test loader ----------
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)
test_loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=256, shuffle=False)


# ---------- Training helpers ----------
def train_local(model, loader, device, epochs=1, lr=1e-3, global_params=None, mu=0.0):
    model.to(device)
    model.train()
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if mu > 0.0 and global_params is not None:
                prox = 0.0
                for p, gp in zip(model.parameters(), global_params):
                    prox += ((p - gp.to(device))**2).sum()
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
    return model

def evaluate_model(model, loader, device):
    model.to(device)
    model.eval()
    ys = []
    ps = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            out = model(xb)
            prob = torch.sigmoid(out).cpu().numpy()
            ps.extend(prob.tolist())
            ys.extend(yb.numpy().tolist())
    ys = np.array(ys)
    ps = np.array(ps)
    preds = (ps >= 0.5).astype(int)
    try:
        auroc = roc_auc_score(ys, ps)
    except:
        auroc = float("nan")
    return {
        "auroc": float(auroc),
        "f1": float(f1_score(ys, preds, zero_division=0)),
        "acc": float(accuracy_score(ys, preds)),
        "precision": float(precision_score(ys, preds, zero_division=0)),
        "recall": float(recall_score(ys, preds, zero_division=0))
    }

# ---------- Federated training loop ----------
def run_federated(algorithm="fedavg"):
    in_features = X_train.shape[1]
    # initialize global model
    global_model = HeartModel(in_features)
    # if FedPer: maintain local heads for each client
    local_heads = [None] * N_CLIENTS

    # initialize local heads from global
    for i in range(N_CLIENTS):
        local_heads[i] = global_model.head.state_dict()

    history = []
    for rnd in range(1, R + 1):
        client_state_dicts = []
        # For FedProx we pass global parameters
        global_params = [p.detach().clone().cpu() for p in global_model.parameters()]

        for i, loader in enumerate(client_loaders):
            # client model from global
            client_model = HeartModel(in_features)
            client_model.load_state_dict(global_model.state_dict())
            # FedPer: restore personalized head
            if algorithm == "fedper" and local_heads[i] is not None:
                cm_sd = client_model.state_dict()
                for k, v in local_heads[i].items():
                    cm_sd["head." + k.split("head.")[1] if k.startswith("head.") else k] = v.clone()
                set_backbone_state_dict(client_model, cm_sd)

            # local training
            mu = PROX_MU if algorithm == "fedprox" else 0.0
            local_epochs = 1 if algorithm == "fedsgd" else LOCAL_EPOCHS
            trained = train_local(client_model, loader, DEVICE, epochs=local_epochs, lr=LR, global_params=global_params, mu=mu)
            client_state_dicts.append(state_dict_to_cpu(trained.state_dict()))
            if algorithm == "fedper":
                local_heads[i] = {k: v.clone() for k, v in trained.state_dict().items() if k.startswith("head.")}

        # aggregate
        if algorithm == "fedper":
            global_sd = fedper_aggregate(client_state_dicts, client_sizes)
        else:
            global_sd = fedavg_aggregate(client_state_dicts, client_sizes)

        global_model.load_state_dict(global_sd)

        # evaluate
        metrics = evaluate_model(global_model, test_loader, DEVICE)
        metrics["round"] = rnd
        history.append(metrics)
        print(f"Round {rnd:03d} | AUROC: {metrics['auroc']:.4f} F1: {metrics['f1']:.4f} ACC: {metrics['acc']:.4f}")
    return global_model, history, local_heads

# ---------- Run experiments ----------
if __name__ == "__main__":
    print("✅ Main block running")
    import time
    import matplotlib.pyplot as plt
    from tabulate import tabulate

    algorithms = ["fedavg", "fedprox", "fedsgd", "fedper"]
    algo_metrics = {}
    results_summary = []

    for algo in algorithms:
        print("\n" + "="*40)
        print("Running:", algo)
        start = time.time()
        gm, hist, heads = run_federated(algorithm=algo)
        elapsed = time.time() - start
        final = hist[-1]

        print(f"Finished {algo} in {elapsed/60:.2f} min | "
              f"Final AUROC: {final['auroc']:.4f}, F1: {final['f1']:.4f}, ACC: {final['acc']:.4f}")

        algo_metrics[algo] = {
            "auroc": [h["auroc"] for h in hist],
            "f1": [h["f1"] for h in hist],
            "accuracy": [h["acc"] for h in hist]
        }

        results_summary.append({
            "Algorithm": algo,
            "Final_AUROC": final["auroc"],
            "Final_F1": final["f1"],
            "Final_ACC": final["acc"],
            "Time_min": elapsed/60.0
        })

    df = pd.DataFrame(results_summary)
    df.to_csv("results.csv", index=False)

    print("\nALL RESULTS SUMMARY:")
    print(tabulate(df, headers="keys", tablefmt="pretty", showindex=False, floatfmt=".4f"))

    # Highlight best
    best_overall_algo = df.loc[((df["Final_AUROC"] + df["Final_F1"] + df["Final_ACC"]) / 3).idxmax()]
    print(f"\n🏆 Best Overall: {best_overall_algo['Algorithm']} "
          f"(AUROC={best_overall_algo['Final_AUROC']:.4f}, "
          f"F1={best_overall_algo['Final_F1']:.4f}, "
          f"ACC={best_overall_algo['Final_ACC']:.4f})")

    # Training curves
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    for algo_name, m in algo_metrics.items():
        plt.plot(m["accuracy"], label=f"{algo_name} Acc")
    plt.xlabel("Rounds"); plt.ylabel("Accuracy"); plt.title("Accuracy Curves"); plt.legend()

    plt.subplot(1, 3, 2)
    for algo_name, m in algo_metrics.items():
        plt.plot(m["f1"], label=f"{algo_name} F1")
    plt.xlabel("Rounds"); plt.ylabel("F1 Score"); plt.title("F1 Curves"); plt.legend()

    plt.subplot(1, 3, 3)
    for algo_name, m in algo_metrics.items():
        plt.plot(m["auroc"], label=f"{algo_name} AUROC")
    plt.xlabel("Rounds"); plt.ylabel("AUROC"); plt.title("AUROC Curves"); plt.legend()

    plt.tight_layout()
    plt.savefig("metrics_curves.png")
    plt.show()

    # Final comparison
    df.plot(x="Algorithm", y=["Final_AUROC", "Final_F1", "Final_ACC"], kind="bar", figsize=(8, 6))
    plt.title("Final Performance Comparison")
    plt.ylabel("Score")
    plt.savefig("final_comparison.png")
    plt.show()
