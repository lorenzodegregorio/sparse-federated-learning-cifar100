"""Lightweight smoke test for zero-sum secure aggregation."""

import torch
from src.secure_aggregation import secure_aggregate

def fedavg(states):
    return {
        key: sum(state[key] for state in states) / len(states)
        for key in states[0]
    }

def main():
    client_states = [
        {"weight": torch.tensor([1.0, 2.0]), "bias": torch.tensor([0.5])},
        {"weight": torch.tensor([2.0, 3.0]), "bias": torch.tensor([1.0])},
        {"weight": torch.tensor([3.0, 4.0]), "bias": torch.tensor([1.5])},
    ]

    plain = fedavg(client_states)
    secure = secure_aggregate(client_states, client_ids=[0, 1, 2])

    for key in plain:
        assert torch.allclose(plain[key], secure[key]), f"Mismatch for {key}"

    print("Secure aggregation smoke test passed.")

if __name__ == "__main__":
    main()
