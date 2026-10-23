"""Model-FLOPs-Utilisation accounting.

MFU = achieved FLOP/s / peak FLOP/s of the device. For a dense decoder the
achieved FLOPs per token are ``6N`` (fwd + bwd over N parameters) plus the
attention term ``12 * L * H * T`` (per token, quadratic in sequence length),
following the PaLM appendix. RL steps additionally spend forward-only passes
for ``old_logp`` and the reference policy (``2N`` each), which we count
separately so the number is honest about where the GPU time goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# dense BF16 peak, no sparsity. Values from vendor datasheets.
_PEAK_TFLOPS = {
    "H100": 989.0,
    "H200": 989.0,
    "H800": 989.0,
    "A100": 312.0,
    "A800": 312.0,
    "L40S": 362.0,
    "A10": 125.0,
    "B200": 2250.0,
    "GB200": 2500.0,
    "MI300X": 1307.0,
}


def peak_flops_for_device(name: str | None = None) -> float:
    """Return peak dense BF16 FLOP/s for the current (or named) accelerator; 1e12 if unknown."""
    if name is None:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
        else:
            return 1e12
    for key, tflops in _PEAK_TFLOPS.items():
        if key.lower() in name.lower():
            return tflops * 1e12
    return 1e12


@dataclass
class MFUCalculator:
    num_params: int
    layers: int
    hidden: int
    peak_flops: float
    world_size: int = 1
    has_reference: bool = True

    def flops_per_token(self, seq_len: int) -> float:
        train = 6.0 * self.num_params + 12.0 * self.layers * self.hidden * seq_len
        # forward-only recompute of old_logp (+ reference) per token
        fwd = 2.0 * self.num_params + 4.0 * self.layers * self.hidden * seq_len
        extra = fwd * (2 if self.has_reference else 1)
        return train + extra

    def mfu(self, tokens: int, seconds: float, seq_len: int) -> float:
        achieved = tokens * self.flops_per_token(seq_len) / max(seconds, 1e-9)
        return achieved / (self.peak_flops * self.world_size)

    @classmethod
    def from_model(
        cls, model: torch.nn.Module, world_size: int = 1, has_reference: bool = True
    ) -> MFUCalculator:
        n = sum(p.numel() for p in model.parameters())
        layers = hidden = 0
        cfg = getattr(model, "cfg", None)
        if cfg is not None and hasattr(cfg, "layers"):
            layers, hidden = cfg.layers, cfg.hidden
        else:
            hf = getattr(getattr(model, "model", None), "config", None)
            if hf is not None:
                layers = int(getattr(hf, "num_hidden_layers", 0))
                hidden = int(getattr(hf, "hidden_size", 0))
        return cls(n, layers, hidden, peak_flops_for_device(), world_size, has_reference)
