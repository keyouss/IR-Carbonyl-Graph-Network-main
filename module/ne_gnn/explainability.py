"""Perturbation-based explanation utilities."""

from __future__ import annotations

import torch


def node_psa_signed_importance(
    model,
    data,
    node_idx: int,
    node_feat_mask: torch.Tensor,
    device: str = "cpu",
    baseline: str = "zero",
):
    """Estimate signed feature importance by perturbing one node feature at a time."""
    model.eval()
    data = data.to(device)
    x = data.x.clone()
    node_feat_mask = node_feat_mask.to(device)

    with torch.no_grad():
        orig_pred = model(x=x, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()

    deltas = torch.zeros_like(node_feat_mask, dtype=torch.float)
    for feature_idx in torch.where(node_feat_mask != 0)[0]:
        x_pert = x.clone()
        if baseline == "zero":
            x_pert[node_idx, feature_idx] = 0
        elif baseline == "mean":
            x_pert[node_idx, feature_idx] = x[:, feature_idx].mean()
        else:
            raise ValueError(f"Unsupported baseline: {baseline}")

        with torch.no_grad():
            pert_pred = model(x=x_pert, edge_index=data.edge_index, edge_attr=data.edge_attr, data=data).item()
        deltas[feature_idx] = orig_pred - pert_pred

    signed_importance = torch.sign(deltas) * node_feat_mask
    return signed_importance, deltas, orig_pred
