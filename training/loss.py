"""
Loss functions for HARP-Pose Dual-Path training.
"""
import torch
import torch.nn as nn


def wrapped_angle_loss(pred: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    """Wrapped L1 for angular difference — gradients at every angle."""
    diff = torch.deg2rad(pred - target)
    diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
    return diff.abs().mean()


def laplace_nll(pred: dict, target: dict,
                lambda_dist: float = 0.01,
                lambda_conf_reg: float = 0.01) -> dict:
    """Laplace NLL: |ε|·√c + ½·log(1/c)."""
    eps_a = torch.deg2rad(pred['heading'] - target['heading'])
    eps_a = (eps_a + torch.pi) % (2 * torch.pi) - torch.pi
    eps_a = eps_a.abs()

    eps_d = (pred['distance'] - target['distance']).abs()

    c = pred['confidence'].clamp(min=1e-4)

    l_angle = (eps_a * c.sqrt()).mean() + 0.5 * c.reciprocal().log().mean()
    l_dist  = (eps_d * c.sqrt()).mean()
    l_reg   = c.log().abs().mean() * lambda_conf_reg

    total = l_angle + lambda_dist * l_dist + l_reg
    return {'total': total, 'angle': l_angle, 'dist': l_dist,
            'conf_reg': l_reg}


def phase1_loss(pred: dict, target: dict) -> dict:
    return laplace_nll(pred, target)


def phase2_loss(pred: dict, target: dict,
                lambda_dist: float = 0.01) -> dict:
    losses = laplace_nll(pred, target, lambda_dist)
    # Auxiliary deep-path loss
    if 'deep_heading' in pred:
        l_a = wrapped_angle_loss(pred['deep_heading'], target['heading'])
        l_d = nn.functional.l1_loss(pred['deep_distance'], target['distance'])
        losses['deep_aux'] = l_a + lambda_dist * l_d
        losses['total'] = losses['total'] + 0.3 * losses['deep_aux']
    return losses


def phase3_loss(pred: dict, target: dict,
                model_state: dict | None = None,
                ewc_state: dict | None = None,
                ewc_lambda: float = 0.0) -> dict:
    losses = laplace_nll(pred, target)
    if ewc_lambda > 0 and ewc_state and model_state:
        l2 = sum(
            ((p - ewc_state[n]).pow(2)).sum()
            for n, p in model_state.items() if n in ewc_state
        )
        losses['ewc'] = ewc_lambda * l2
        losses['total'] = losses['total'] + losses['ewc']
    return losses
