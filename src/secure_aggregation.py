import torch

def secure_aggregate(client_states, client_ids, seed_offset=0):
    """
    Secure aggregation with zero-sum noise masking.

    Args:
        client_states: List of state_dicts from clients (each a dict of param_name->Tensor).
        client_ids:    List of client IDs in the same order as client_states.
        seed_offset:   Optional int to offset random seed for reproducibility.

    Returns:
        agg_state: Aggregated state_dict with zero-sum noise removed.
    """
    num_clients = len(client_states)
    # 1) Generate noise for first N-1 clients, accumulate sum
    noise_sum = {}
    noise_dicts = []
    for idx in range(num_clients - 1):
        cid = client_ids[idx]
        torch.manual_seed(seed_offset + cid)
        state = client_states[idx]
        noise = {k: torch.randn_like(v) for k, v in state.items()}
        noise_dicts.append(noise)
        for k, v in noise.items():
            noise_sum[k] = noise_sum.get(k, torch.zeros_like(v)) + v
    # 2) Last client's noise is negative of accumulated sum
    last_noise = {k: -v for k, v in noise_sum.items()}
    noise_dicts.append(last_noise)
    # 3) Apply noise to each client's state
    masked_states = []
    for state, noise in zip(client_states, noise_dicts):
        masked = {k: state[k] + noise[k] for k in state}
        masked_states.append(masked)
    # 4) Aggregate masked states (simple average)
    agg_state = {}
    for k in masked_states[0]:
        agg_state[k] = sum(ms[k] for ms in masked_states) / num_clients
    return agg_state
