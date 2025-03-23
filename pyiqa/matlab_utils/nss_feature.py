import torch
from typing import Tuple


def estimate_aggd_param(
    block: torch.Tensor, return_sigma=False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    Args:
        block (Tensor): Image block with shape (b, 1, h, w).
    Returns:
        Tensor: alpha, beta_l and beta_r for the AGGD distribution
        (Estimating the parames in Equation 7 in the paper).
    """
    gam = torch.arange(0.2, 10 + 0.001, 0.001).to(block)
    r_gam = (
        2 * torch.lgamma(2.0 / gam)
        - (torch.lgamma(1.0 / gam) + torch.lgamma(3.0 / gam))
    ).exp()
    r_gam = r_gam.repeat(block.shape[0], 1)

    mask_left = block < 0
    mask_right = block > 0
    count_left = mask_left.sum(dim=(-1, -2), dtype=torch.float32)
    count_right = mask_right.sum(dim=(-1, -2), dtype=torch.float32)

    left_std = torch.sqrt((block * mask_left).pow(2).sum(dim=(-1, -2)) / (count_left))
    right_std = torch.sqrt(
        (block * mask_right).pow(2).sum(dim=(-1, -2)) / (count_right)
    )

    gammahat = left_std / right_std
    rhat = block.abs().mean(dim=(-1, -2)).pow(2) / block.pow(2).mean(dim=(-1, -2))
    rhatnorm = (rhat * (gammahat.pow(3) + 1) * (gammahat + 1)) / (
        gammahat.pow(2) + 1
    ).pow(2)
    array_position = (r_gam - rhatnorm).abs().argmin(dim=-1)

    alpha = gam[array_position]
    beta_l = (
        left_std.squeeze(-1)
        * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    )
    beta_r = (
        right_std.squeeze(-1)
        * (torch.lgamma(1 / alpha) - torch.lgamma(3 / alpha)).exp().sqrt()
    )

    if return_sigma:
        return alpha, left_std.squeeze(-1), right_std.squeeze(-1)
    else:
        return alpha, beta_l, beta_r


def compute_nss_features(luma_nrmlzd: torch.Tensor) -> torch.Tensor:
    alpha, betal, betar = estimate_aggd_param(luma_nrmlzd, return_sigma=False)
    features = [alpha, (betal + betar) / 2]

    shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]

    for shift in shifts:
        shifted_luma_nrmlzd = torch.roll(luma_nrmlzd, shifts=shift, dims=(-2, -1))
        alpha, betal, betar = estimate_aggd_param(
            luma_nrmlzd * shifted_luma_nrmlzd, return_sigma=False
        )
        distmean = (betar - betal) * torch.exp(
            torch.lgamma(2 / alpha) - torch.lgamma(1 / alpha)
        )
        features.extend((alpha, distmean, betal, betar))

    return torch.stack(features, dim=-1)
