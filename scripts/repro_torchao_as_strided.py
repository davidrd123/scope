#!/usr/bin/env python3
"""
Attempt at minimal repro for torch.compile + torchao.quantization.Float8Tensor
aten.as_strided.default dispatch gap.

NOTE: This script is not a reliable standalone repro. In our toy MLPs (even with
explicit view/reshape/transpose ops), we did not hit the failure; we hit it in
the full transformer-based diffusion pipeline.

Current hypothesis: `aten.as_strided` is being introduced by AOTAutograd
aliasing/stride-correction logic (e.g., `gen_alias_from_base`) during
compilation, not necessarily by explicit model `.as_strided(...)` calls.

Canonical repro that DOES trigger the issue:
    SCOPE_KV_BIAS_BACKEND=fa4 \\
    .venv-b300-cu130-decode/bin/python scripts/profile_krea_pipeline_blocks.py \\
      --quantization fp8_e4m3fn --compile

See: notes/issues/torchao-as-strided-dispatch.md

As of 2025-12-26: `scripts/patch_float8_as_strided.py` provides a PerTensor-only
monkeypatch that unblocks `--compile + fp8` for experiments (upstream support is
still missing).

Environment:
    pip install torch torchao
    # Tested with torch 2.9.0+cu130, torchao 0.15.0
"""

import torch
import torch.nn as nn

# Check versions
print(f"torch: {torch.__version__}")
try:
    import torchao
    print(f"torchao: {torchao.__version__}")
except Exception as e:
    print(f"torchao import failed: {e}")
    raise

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig
from torchao.quantization.granularity import PerTensor


class SimpleModel(nn.Module):
    """Minimal model with a Linear layer."""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x


class ModelWithViews(nn.Module):
    """Model with view/reshape ops (often *doesn't* trigger the issue)."""
    def __init__(self, dim: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 2)
        self.linear2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, dim]
        x = self.linear1(x)  # [B, dim*2]
        # Reshape to trigger view ops
        B = x.shape[0]
        x = x.view(B, 2, -1)  # [B, 2, dim]
        x = x.transpose(1, 2)  # [B, dim, 2]
        x = x.reshape(B, -1)  # [B, dim*2]
        # Slice back to dim
        x = x[:, :256]  # [B, dim]
        x = self.linear2(x)
        return x


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["simple", "views"], default="views",
                        help="Which model to use")
    parser.add_argument("--mode", default=None,
                        help="torch.compile mode (default, reduce-overhead, max-autotune)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # 1. Create model
    if args.model == "simple":
        model = SimpleModel(dim=256).to(device).to(torch.bfloat16)
        print("Created SimpleModel")
    else:
        model = ModelWithViews(dim=256).to(device).to(torch.bfloat16)
        print("Created ModelWithViews (view/reshape/transpose ops)")

    # 2. Quantize with Float8DynamicActivationFloat8WeightConfig (per-tensor scale)
    #    This produces torchao.quantization.Float8Tensor
    quantize_(
        model,
        Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor()),
    )
    print("Quantized model with Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())")

    # 3. Compile
    #
    # Note: the `as_strided` that trips `torchao.quantization.Float8Tensor` dispatch can come
    # from AOTAutograd aliasing logic (gen_alias_from_base), not necessarily model-level view ops.
    compile_kwargs = {}
    if args.mode:
        compile_kwargs["mode"] = args.mode
    compiled_model = torch.compile(model, **compile_kwargs)
    print(f"Compiled model with torch.compile(mode={args.mode})")

    # 4. Run inference (best-effort)
    x = torch.randn(1, 256, device=device, dtype=torch.bfloat16)
    print("Running inference...")

    try:
        # Avoid inference_mode here so we don't conflate this with the separate
        # tensor-subclass + inference_mode version_counter failure family.
        with torch.no_grad():
            out = compiled_model(x)
        print(f"Success! Output shape: {out.shape}")
        print("NOTE: This script often won't reproduce; use the canonical pipeline repro instead.")
    except NotImplementedError as e:
        if "aten.as_strided" in str(e):
            print("\n" + "=" * 60)
            print("REPRO CONFIRMED: aten.as_strided.default dispatch missing")
            print("=" * 60)
            print(f"\nError: {e}")
            raise
        else:
            raise


if __name__ == "__main__":
    main()
