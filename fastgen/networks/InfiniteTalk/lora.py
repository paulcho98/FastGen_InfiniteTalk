"""LoRA utilities for InfiniteTalk FastGen adaptation.

Provides parameter-efficient training for 14B models via Low-Rank Adaptation.
Used by both the student (causal) and fake_score (bidirectional) networks.

InfiniteTalk uses a single 14B model for all three roles (teacher, student,
fake_score). LoRA adapters make training feasible by only updating ~0.5-1GB
of parameters instead of the full 14B:
  - Teacher: LoRA merged into base weights (frozen, no runtime overhead)
  - Student / fake_score: runtime LoRA adapters (trainable)

Module layout (WanModel):
  - blocks.{i}.self_attn.{q,k,v,o}         -- self-attention projections
  - blocks.{i}.cross_attn.{q,k,v,o}        -- text cross-attention (inherited)
  - blocks.{i}.cross_attn.{k_img,v_img}    -- image cross-attention
  - blocks.{i}.ffn.{0,2}                   -- feed-forward linears
  - blocks.{i}.audio_cross_attn.*           -- audio (excluded from LoRA)
  - patch_embedding, text_embedding, time_embedding, time_projection,
    head, img_emb, audio_proj               -- stem / head (excluded from LoRA)
"""

import math
import logging
from typing import Dict, List, Optional, Set, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default target / exclusion patterns
# ---------------------------------------------------------------------------

# Module name fragments that identify eligible linear layers inside
# transformer blocks.  Matched against the *full dotted name* of the child
# nn.Linear relative to the model root (e.g. "blocks.0.self_attn.q").
DEFAULT_TARGET_PATTERNS: Tuple[str, ...] = (
    "self_attn.q",
    "self_attn.k",
    "self_attn.v",
    "self_attn.o",
    "cross_attn.q",
    "cross_attn.k",      # text k (inherited from WanSelfAttention)
    "cross_attn.v",      # text v (inherited from WanSelfAttention)
    "cross_attn.k_img",
    "cross_attn.v_img",
    "cross_attn.o",
    "ffn.0",
    "ffn.2",
)

# Top-level modules that must never receive LoRA adapters.
EXCLUDE_PATTERNS: Tuple[str, ...] = (
    "patch_embedding",
    "time_embedding",
    "text_embedding",
    "time_projection",
    "head",
    "img_emb",
    "audio_proj",
    "audio_cross_attn",
)


# ---------------------------------------------------------------------------
# LoRALinear — drop-in replacement for nn.Linear with LoRA adapters
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Linear layer augmented with a low-rank adapter (LoRA).

    Wraps an existing ``nn.Linear`` (the *base*) and adds trainable low-rank
    matrices A and B such that the effective weight becomes::

        W_eff = W_base + scaling * B @ A

    where ``scaling = alpha / rank``.

    During training the base weight is typically frozen; only A and B receive
    gradients.  For inference the adapter can be merged into the base weight
    via :func:`merge_lora` to eliminate runtime overhead.
    """

    def __init__(self, base_linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.base = base_linear  # original linear — will be frozen externally
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA adapter matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # Standard LoRA initialisation: A ~ Kaiming uniform, B = 0
        # so the adapter contributes nothing at init (delta = B @ A = 0).
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self) -> torch.Tensor:
        """Effective weight (base + LoRA delta).  Read-only convenience."""
        return self.base.weight + self.scaling * (self.lora_B @ self.lora_A)

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return self.base.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward (no grad through base weights when frozen)
        base_out = F.linear(x, self.base.weight, self.base.bias)
        # LoRA forward: x @ A^T @ B^T * scaling
        lora_out = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base_out + lora_out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.base.in_features}, "
            f"out_features={self.base.out_features}, "
            f"rank={self.rank}, alpha={self.alpha}, "
            f"bias={self.base.bias is not None}"
        )


# ---------------------------------------------------------------------------
# apply_lora — inject LoRA adapters into a model
# ---------------------------------------------------------------------------

def _matches_target(full_name: str, targets: Tuple[str, ...]) -> bool:
    """Return True if *full_name* contains any of the target patterns."""
    return any(t in full_name for t in targets)


def _matches_exclude(full_name: str, excludes: Tuple[str, ...]) -> bool:
    """Return True if *full_name* contains any exclusion pattern."""
    return any(e in full_name for e in excludes)


def apply_lora(
    model: nn.Module,
    rank: int,
    alpha: float,
    target_modules: Optional[Tuple[str, ...]] = None,
    exclude_modules: Optional[Tuple[str, ...]] = None,
) -> nn.Module:
    """Replace eligible ``nn.Linear`` layers with :class:`LoRALinear`.

    Traverses the model and replaces each ``nn.Linear`` whose full dotted
    name matches a target pattern (and does not match any exclusion pattern)
    with a ``LoRALinear`` wrapper.

    Args:
        model: The model to modify (mutated **in place**).
        rank: LoRA rank (e.g. 128).
        alpha: LoRA scaling alpha (e.g. 64).
        target_modules: Name fragments to target.  Defaults to
            :data:`DEFAULT_TARGET_PATTERNS`.
        exclude_modules: Name fragments to exclude.  Defaults to
            :data:`EXCLUDE_PATTERNS`.

    Returns:
        The same model instance (modified in place).
    """
    if target_modules is None:
        target_modules = DEFAULT_TARGET_PATTERNS
    if exclude_modules is None:
        exclude_modules = EXCLUDE_PATTERNS

    replaced = 0

    # Build a list of (parent_module, attr_name, full_name) for replacement.
    # We iterate named_modules and inspect direct children to avoid
    # mutating the iterator while traversing.
    replacements: List[Tuple[nn.Module, str, str]] = []
    for parent_name, parent_module in model.named_modules():
        for child_name, child_module in parent_module.named_children():
            if not isinstance(child_module, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if _matches_exclude(full_name, exclude_modules):
                continue
            if _matches_target(full_name, target_modules):
                replacements.append((parent_module, child_name, full_name))

    for parent_module, child_name, full_name in replacements:
        original_linear = getattr(parent_module, child_name)
        lora_linear = LoRALinear(original_linear, rank=rank, alpha=alpha)
        setattr(parent_module, child_name, lora_linear)
        replaced += 1

    logger.info(
        f"[apply_lora] Replaced {replaced} nn.Linear layers with LoRALinear "
        f"(rank={rank}, alpha={alpha})"
    )
    return model


# ---------------------------------------------------------------------------
# freeze_base — freeze everything except LoRA + audio
# ---------------------------------------------------------------------------

def freeze_base(model: nn.Module) -> None:
    """Freeze all parameters, then unfreeze LoRA adapters and audio modules.

    After calling this function:
      - Base transformer weights: frozen (requires_grad=False)
      - ``lora_A`` / ``lora_B`` parameters: trainable
      - ``audio_proj`` / ``audio_cross_attn`` parameters: trainable
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False

    # 2. Unfreeze LoRA adapters
    lora_count = 0
    for name, p in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            p.requires_grad = True
            lora_count += 1

    # 3. Unfreeze audio modules (AudioProjModel + SingleStreamAttention)
    audio_count = 0
    for name, p in model.named_parameters():
        if "audio_proj" in name or "audio_cross_attn" in name:
            p.requires_grad = True
            audio_count += 1

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[freeze_base] Trainable: {trainable_params:,} / {total_params:,} params "
        f"({trainable_params / total_params * 100:.2f}%) — "
        f"{lora_count} LoRA params, {audio_count} audio params"
    )


# ---------------------------------------------------------------------------
# merge_lora — fold LoRA weights into base (for frozen teacher)
# ---------------------------------------------------------------------------

def merge_lora(model: nn.Module) -> int:
    """Merge all LoRA adapters into their base linear layers in-place.

    After merging, the ``LoRALinear`` modules still exist but their adapters
    have been absorbed into ``base.weight``.  Typically followed by replacing
    the ``LoRALinear`` wrappers back with plain ``nn.Linear`` or simply
    freezing the entire model.

    Returns:
        Number of LoRA modules merged.
    """
    merged = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            # delta = scaling * B @ A  (same shapes as base.weight)
            delta = module.scaling * (module.lora_B @ module.lora_A)
            module.base.weight.data.add_(delta.to(module.base.weight.dtype))
            # Reset adapter to zero so it contributes nothing if forward is called
            module.lora_B.data.zero_()
            merged += 1

    logger.info(f"[merge_lora] Merged {merged} LoRA adapters into base weights")
    return merged


# ---------------------------------------------------------------------------
# merge_lora_from_file — merge from an external state dict (InfiniteTalk fmt)
# ---------------------------------------------------------------------------

def merge_lora_from_file(
    model: nn.Module,
    lora_sd: Dict[str, torch.Tensor],
    lora_rank: int,
    lora_alpha: float,
    prefix: str = "diffusion_model.",
) -> int:
    """Merge LoRA weights from a state dict into the base model in-place.

    Handles the InfiniteTalk / Wan checkpoint key conventions:

    * **Standard LoRA**:
      ``{prefix}{module}.lora_down.weight`` (A) +
      ``{prefix}{module}.lora_up.weight`` (B).
      Merge: ``W += (alpha / rank) * B @ A``

    * **Weight diffs**:
      ``{prefix}{module}.diff`` — added directly to ``weight``.

    * **Bias diffs**:
      ``{prefix}{module}.diff_b`` — added directly to ``bias``.

    All matrix multiplications are performed on GPU when available for speed
    (14B LoRA merge takes 40+ minutes on CPU, ~30 seconds on GPU).

    Args:
        model: Target model whose parameters will be modified.
        lora_sd: State dict loaded from a LoRA checkpoint file.
        lora_rank: Rank used when the LoRA was trained.
        lora_alpha: Alpha used when the LoRA was trained.
        prefix: Key prefix to strip (default ``"diffusion_model."``).

    Returns:
        Number of parameter updates applied.
    """
    scaling = lora_alpha / lora_rank
    merge_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- Collect LoRA pairs and weight/bias diffs ---
    lora_pairs: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}  # base_key -> (A, B)
    weight_diffs: Dict[str, torch.Tensor] = {}   # base_key -> delta tensor
    bias_diffs: Dict[str, torch.Tensor] = {}     # base_key -> delta tensor

    for key, tensor in lora_sd.items():
        if not key.startswith(prefix):
            continue
        stripped = key[len(prefix):]

        if stripped.endswith("lora_down.weight"):
            base_key = stripped.replace("lora_down.weight", "weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = [None, None]
            lora_pairs[base_key][0] = tensor  # A (down)
        elif stripped.endswith("lora_up.weight"):
            base_key = stripped.replace("lora_up.weight", "weight")
            if base_key not in lora_pairs:
                lora_pairs[base_key] = [None, None]
            lora_pairs[base_key][1] = tensor  # B (up)
        elif stripped.endswith("diff_b"):
            base_key = stripped.replace("diff_b", "bias")
            bias_diffs[base_key] = tensor
        elif stripped.endswith("diff"):
            base_key = stripped.replace("diff", "weight")
            weight_diffs[base_key] = tensor

    # --- Helper to resolve dotted name to parameter ---
    model_params = dict(model.named_parameters())

    applied = 0

    # Merge LoRA pairs: W += scaling * B @ A
    for base_key, (A, B) in lora_pairs.items():
        if A is None or B is None:
            logger.warning(f"[merge_lora_from_file] Incomplete LoRA pair for {base_key}, skipping")
            continue
        if base_key not in model_params:
            logger.warning(f"[merge_lora_from_file] Parameter {base_key} not found in model, skipping")
            continue

        param = model_params[base_key]
        delta = scaling * (
            B.to(device=merge_device, dtype=param.dtype)
            @ A.to(device=merge_device, dtype=param.dtype)
        )
        param.data.add_(delta.to(param.device))
        applied += 1

    # Merge weight diffs: W += diff
    for base_key, diff in weight_diffs.items():
        if base_key not in model_params:
            logger.warning(f"[merge_lora_from_file] Parameter {base_key} not found (weight diff), skipping")
            continue
        param = model_params[base_key]
        param.data.add_(diff.to(device=param.device, dtype=param.dtype))
        applied += 1

    # Merge bias diffs: b += diff_b
    for base_key, diff in bias_diffs.items():
        if base_key not in model_params:
            logger.warning(f"[merge_lora_from_file] Parameter {base_key} not found (bias diff), skipping")
            continue
        param = model_params[base_key]
        param.data.add_(diff.to(device=param.device, dtype=param.dtype))
        applied += 1

    logger.info(
        f"[merge_lora_from_file] Applied {applied} updates "
        f"({len(lora_pairs)} LoRA pairs, {len(weight_diffs)} weight diffs, "
        f"{len(bias_diffs)} bias diffs) with scaling={scaling}"
    )
    return applied


# ---------------------------------------------------------------------------
# extract_lora_state_dict — lightweight checkpoint saving
# ---------------------------------------------------------------------------

def extract_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA A/B parameters for lightweight checkpoint saving.

    Returns a state dict containing only the ``lora_A`` and ``lora_B``
    parameters, suitable for ``torch.save()``.  Typical size for a 14B
    model with rank=128 is ~0.5-1 GB (vs ~28 GB for the full model).
    """
    lora_sd: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_sd[name] = param.data.clone()
    logger.info(f"[extract_lora_state_dict] Extracted {len(lora_sd)} LoRA tensors")
    return lora_sd


# ---------------------------------------------------------------------------
# load_lora_state_dict — restore LoRA params into existing LoRALinear layers
# ---------------------------------------------------------------------------

def load_lora_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Load LoRA parameters into existing ``LoRALinear`` layers.

    Args:
        model: Model with ``LoRALinear`` modules already injected via
            :func:`apply_lora`.
        state_dict: Dict mapping parameter names to tensors (as returned
            by :func:`extract_lora_state_dict`).
        strict: If True (default), raise ``RuntimeError`` when there are
            missing or unexpected keys.

    Returns:
        Tuple of (loaded_keys, missing_keys).
    """
    model_lora_params = {
        name: param
        for name, param in model.named_parameters()
        if "lora_A" in name or "lora_B" in name
    }

    loaded: List[str] = []
    missing: List[str] = []
    unexpected: List[str] = []

    for name in model_lora_params:
        if name in state_dict:
            model_lora_params[name].data.copy_(
                state_dict[name].to(model_lora_params[name].device)
            )
            loaded.append(name)
        else:
            missing.append(name)

    for name in state_dict:
        if name not in model_lora_params:
            unexpected.append(name)

    if strict and (missing or unexpected):
        raise RuntimeError(
            f"[load_lora_state_dict] strict=True but found "
            f"{len(missing)} missing and {len(unexpected)} unexpected keys.\n"
            f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"  Unexpected: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}"
        )

    logger.info(
        f"[load_lora_state_dict] Loaded {len(loaded)} params, "
        f"{len(missing)} missing, {len(unexpected)} unexpected"
    )
    return loaded, missing
