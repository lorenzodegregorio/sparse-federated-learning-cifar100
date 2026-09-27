import torch
import random

def compute_sensitivity(model):
    """
    Computes sensitivity of model parameters based on absolute gradients.
    Returns:
        sensitivity (dict): Parameter name → tensor of absolute gradient values.
    """
    sensitivity = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            sensitivity[name] = param.grad.abs().detach().clone()
    return sensitivity

def calibrate_gradient_mask(model, strategy='least-sensitive', sparsity=0.5, fisher_info=None):
    """
    Creates pruning masks for a given model based on a sparsity strategy.
    
    Args:
        model: The model to prune.
        strategy: Strategy name. One of:
            ['least-sensitive', 'most-sensitive', 'lowest-magnitude', 'highest-magnitude', 'random']
        sparsity: Fraction of weights to zero out (e.g., 0.5 = 50% sparsity).
        fisher_info: Optional precomputed sensitivity info.
    
    Returns:
        masks: A list of masks (same shape as parameters) to apply.
    """
    if fisher_info is not None:
        sensitivity = fisher_info
    else:
        sensitivity = compute_sensitivity(model)

    masks = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if strategy in ['least-sensitive', 'most-sensitive']:
            if name not in sensitivity:
                print(f"⚠️ Skipping parameter '{name}' (no sensitivity data)")
                masks.append(torch.ones_like(param))  # Keep weights unchanged
                continue

            sens = sensitivity[name]
            k = int(param.numel() * (1 - sparsity))

            if strategy == 'least-sensitive':
                threshold = torch.topk(sens.view(-1), k, largest=False).values[-1]
                mask = (sens <= threshold).float()
            else:  # most-sensitive
                threshold = torch.topk(sens.view(-1), k, largest=True).values[-1]
                mask = (sens >= threshold).float()

            masks.append(mask.view_as(param))

        elif strategy == 'lowest-magnitude':
            k = int(param.numel() * (1 - sparsity))
            threshold = torch.topk(param.abs().view(-1), k, largest=False).values[-1]
            mask = (param.abs() >= threshold).float()
            masks.append(mask.view_as(param))

        elif strategy == 'highest-magnitude':
            k = int(param.numel() * (1 - sparsity))
            threshold = torch.topk(param.abs().view(-1), k, largest=True).values[-1]
            mask = (param.abs() >= threshold).float()
            masks.append(mask.view_as(param))

        elif strategy == 'random':
            mask = (torch.rand_like(param) < (1 - sparsity)).float()
            masks.append(mask)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    return masks

def apply_gradient_mask(model, mask):
    """
    Applies a precomputed binary gradient mask to the model's parameters.

    Args:
        model: The model being trained.
        mask: A dictionary or list of tensors with the same structure as model.parameters().
              Each tensor is a binary mask (0 = masked, 1 = keep).
    """
    with torch.no_grad():
        for (name, param), mask_tensor in zip(model.named_parameters(), mask):
            if param.grad is not None:
                param.grad *= mask_tensor.to(param.device)

class SparseSGD(torch.optim.SGD):
    """
    SGD optimizer that applies sparsity masks to gradients during updates.
    """
    def __init__(self, params, lr, mask=None):
        super(SparseSGD, self).__init__(params, lr=lr)
        self.mask = mask  # List of masks for each parameter (same order)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if self.mask is not None and i < len(self.mask):
                    d_p.mul_(self.mask[i])  # Apply mask to gradients
                p.data.add_(-group['lr'], d_p)
        
        return loss