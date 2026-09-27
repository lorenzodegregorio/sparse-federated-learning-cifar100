# Results Summary

This file summarizes the final test accuracies obtained in the public project results.

## Final test accuracy

| Model | Final test accuracy (%) |
|---|---:|
| Centralized | 73.27 |
| Sparse fine-tuned | 70.66 |
| FedAvg baseline, IID | 72.12 |
| Federated sparse, IID | 70.66 |
| FedAvg baseline, Non-IID Nc=1 | 63.41 |
| Federated sparse, Non-IID Nc=1 | 67.02 |
| FedAvg baseline, Non-IID Nc=5 | 70.31 |
| Federated sparse, Non-IID Nc=5 | 70.38 |
| FedAvg baseline, Non-IID Nc=10 | 70.75 |
| Federated sparse, Non-IID Nc=10 | 71.57 |
| FedAvg baseline, Non-IID Nc=50 | 72.54 |
| Federated sparse, Non-IID Nc=50 | 73.08 |

## Main observations

- The centralized model achieved the highest final test accuracy among the main baseline settings.
- In the IID federated setting, sparse federated learning remained close to the FedAvg baseline.
- In the non-IID experiments, performance generally improved as the number of classes per client increased, meaning that the client data distribution became less heterogeneous.
- The sparse federated variant achieved competitive performance in several non-IID settings, suggesting that sparse updates can reduce update density while retaining useful predictive performance.

## Notes

The results are included as lightweight aggregated metrics only. Model checkpoints, raw training logs and downloaded datasets are not included in the public repository.
