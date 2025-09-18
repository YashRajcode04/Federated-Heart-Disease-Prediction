# fl_algorithms.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Model: small MLP with separable backbone/head for FedPer
# ---------------------------
class HeartModel(nn.Module):
    def __init__(self, in_features, hidden=64, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        # head = personalization layer (local in FedPer)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        out = self.head(x)
        return out.squeeze(1)  # logits (before sigmoid)


# ---------------------------
# Utilities to split/merge state_dict
# ---------------------------
def get_backbone_state_dict(state_dict):
    """Return only backbone weights (exclude head.* for FedPer)."""
    return {k: v.clone() for k, v in state_dict.items() if not k.startswith("head.")}

def set_backbone_state_dict(model, backbone_state_dict):
    """Load only backbone weights into model (leave head as is)."""
    m_sd = model.state_dict()
    for k, v in backbone_state_dict.items():
        if k in m_sd:
            m_sd[k] = v.clone()
    model.load_state_dict(m_sd)


def average_state_dicts(state_dicts, weights=None):
    """Average list of state_dicts with optional weights."""
    if weights is None:
        weights = [1.0 / len(state_dicts)] * len(state_dicts)
    avg = {}
    for k in state_dicts[0].keys():
        avg_k = torch.zeros_like(state_dicts[0][k], dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            avg_k += w * sd[k].float()
        avg[k] = avg_k.clone()
    return avg


# ---------------------------
# FedAvg aggregator
# ---------------------------
def fedavg_aggregate(client_state_dicts, client_sizes=None):
    """Federated Averaging aggregation."""
    if client_sizes is None:
        return average_state_dicts(client_state_dicts)
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    return average_state_dicts(client_state_dicts, weights)


# ---------------------------
# FedPer aggregator: average only backbone (keep heads local)
# ---------------------------
def fedper_aggregate(client_state_dicts, client_sizes=None):
    """Aggregate only backbone weights (heads stay local)."""
    backbones = [get_backbone_state_dict(sd) for sd in client_state_dicts]
    if client_sizes is None:
        avg_backbone = average_state_dicts(backbones)
    else:
        total = sum(client_sizes)
        weights = [s / total for s in client_sizes]
        avg_backbone = average_state_dicts(backbones, weights)

    # Copy one state_dict and replace backbone with averaged version
    global_sd = copy.deepcopy(client_state_dicts[0])
    for k, v in avg_backbone.items():
        global_sd[k] = v.clone()
    return global_sd


# ---------------------------
# FedSGD: no special aggregator (just FedAvg with local_epochs=1)
# ---------------------------


# ---------------------------
# Helper: copy model state_dict (to CPU tensors)
# ---------------------------
def state_dict_to_cpu(sd):
    """Clone state_dict to CPU to avoid GPU reference issues."""
    return {k: v.detach().cpu().clone() for k, v in sd.items()}
