"""KL-divergence estimators and the adaptive KL coefficient controller.

Given sampled-token log-probs under the current policy ``logp`` and the
reference ``ref_logp`` (both ``[B, T]``), with ``d = ref_logp - logp``:

* ``k1``:  ``-d``                      unbiased, high variance, can be negative
* ``k2``:  ``d^2 / 2``                 biased, low variance, always >= 0
* ``k3``:  ``exp(d) - d - 1``          unbiased *and* low variance (Schulman 2020)

``k3`` is the default in GRPO-style training. All estimators are clamped on
the exponent to keep a single wildly off-policy token from producing an
``inf`` that nukes the step.
"""

from __future__ import annotations

import torch


def kl_penalty(
    logp: torch.Tensor, ref_logp: torch.Tensor, estimator: str = "k3", clamp: float = 20.0
) -> torch.Tensor:
    d = (ref_logp - logp).clamp(-clamp, clamp)
    if estimator == "k1":
        return -d
    if estimator == "k2":
        return 0.5 * d * d
    if estimator == "k3":
        return torch.exp(d) - d - 1.0
    raise ValueError(f"unknown KL estimator {estimator!r}")


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    m = mask.to(x.dtype)
    if dim is None:
        return (x * m).sum() / m.sum().clamp_min(1.0)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1.0)


class AdaptiveKLController:
    """Ziegler et al. (2019) proportional controller on the KL coefficient.

    If measured KL exceeds the target the coefficient grows, otherwise it
    shrinks, with the per-step change bounded to +/-20% so the controller
    cannot oscillate violently.
    """

    def __init__(self, init_coef: float, target: float | None, horizon: int = 10_000):
        self.coef = init_coef
        self.target = target
        self.horizon = max(horizon, 1)

    def update(self, current_kl: float, n_steps: int = 1) -> float:
        if self.target is None:
            return self.coef
        err = max(min(current_kl / self.target - 1.0, 0.2), -0.2)
        mult = 1.0 + err * n_steps / self.horizon
        self.coef *= mult
        return self.coef
