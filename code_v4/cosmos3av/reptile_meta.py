"""Reptile theta bookkeeping and inner-optimizer helpers (torch side; FSDP/DTensor aware).

The model's own (FSDP-sharded, fp32 master) parameters are the fast weights of the inner loop, so the
ordinary Cosmos training stack runs it unchanged. :class:`ReptileMetaState` keeps ONE extra copy of every
trainable parameter's local shard as ``theta`` and performs::

    theta <- theta + eps * d,   d = theta_tilde - theta   (or the weighted mean over several tasks)
    model <- theta

Only parameters with ``requires_grad`` after the optimizer was built are part of theta (the Cosmos optimizer
builder freezes everything outside ``optimizer.keys_to_select``), i.e. theta == the trainable set of the
Stage 1 / Stage 2 recipe. Nothing here depends on the Cosmos framework except the optional gradient-clip
helpers, which are imported lazily.
"""
from __future__ import annotations

import math
from typing import Any, Iterable

import torch
import torch.distributed as dist
from torch import nn

GROUP_ACTION_HEADS = "action_heads"
_ACTION_HEAD_TOKENS = ("action2llm", "llm2action", "action_modality_embed", "action_pos_embed")
_NAMED_GROUPS = ("moe_gen", "time_embedder", "vae2llm", "llm2vae", "k_norm_und_for_gen")


def canonical_param_name(name: str) -> str:
    """Strip activation-checkpoint / torch.compile wrapper segments from a parameter name."""
    return name.replace("_orig_mod.", "").replace("_checkpoint_wrapped_module.", "")


def param_group(name: str) -> str:
    """Diagnostic group of a (canonical) parameter name."""
    if any(tok in name for tok in _ACTION_HEAD_TOKENS):
        return GROUP_ACTION_HEADS
    if "lora_" in name:
        return "lora"
    for g in _NAMED_GROUPS:
        if g in name:
            return g
    return "other"


def net_param_lookup(net: nn.Module) -> dict[str, nn.Parameter]:
    """``{canonical name: parameter}`` over ``net.named_parameters()``."""
    out: dict[str, nn.Parameter] = {}
    for name, p in net.named_parameters():
        c = canonical_param_name(name)
        if c in out and out[c] is not p:
            raise RuntimeError(f"Two distinct parameters canonicalize to {c!r} ({name!r})")
        out[c] = p
    return out


def local_tensor(t: torch.Tensor) -> torch.Tensor:
    """The rank-local shard of a DTensor (identity for plain tensors)."""
    try:
        from torch.distributed.tensor import DTensor

        if isinstance(t, DTensor):
            return t.to_local()
    except Exception:  # noqa: BLE001
        pass
    return t


# --------------------------------------------------------------------------------------------
# inner optimizer helpers
# --------------------------------------------------------------------------------------------
def _optimizers(container: Any) -> list[torch.optim.Optimizer]:
    return list(getattr(container, "optimizers", None) or [container])


def reset_optimizer_state(container: Any) -> None:
    """Forget the inner optimizer's moments (and FusedAdam's per-group step counter) between tasks."""
    for opt in _optimizers(container):
        opt.state.clear()
        for group in opt.param_groups:
            group.pop("step", None)


def capture_base_lrs(container: Any) -> list[list[float]]:
    """Per-group base LRs (``lr * multiplier``). ``initial_lr`` when a LambdaLR already touched the groups
    (the recipe's scheduler is built but never stepped by the Reptile trainer)."""
    return [[float(g.get("initial_lr", g["lr"])) for g in opt.param_groups] for opt in _optimizers(container)]


def set_lr_scale(container: Any, base_lrs: list[list[float]], scale: float) -> None:
    for opt, bases in zip(_optimizers(container), base_lrs, strict=True):
        for group, base in zip(opt.param_groups, bases, strict=True):
            group["lr"] = base * float(scale)


def clip_grads(params: Iterable[torch.nn.Parameter], max_norm: float | None, force_finite: bool = True) -> float:
    """Global-norm gradient clipping with the framework's mesh-aware helpers (the same math as the
    ``GradClip`` callback of the baseline: NaN/Inf -> 0 when the norm is non-finite, then one global rescale).
    Returns the pre-clip global norm (nan when no parameter has a gradient)."""
    from cosmos_framework.callbacks.grad_clip import (
        _clip_grads_with_global_norm,
        _fused_nan_to_num,
        _group_params_by_mesh,
        _total_norm_by_mesh,
    )

    with_grad = [p for p in params if p.grad is not None]
    if not with_grad:
        return float("nan")
    groups = _group_params_by_mesh(with_grad)
    total, _ = _total_norm_by_mesh(groups, error_if_nonfinite=False, foreach=True)
    if force_finite and not bool(torch.isfinite(total)):
        _fused_nan_to_num([p.grad for ps in groups.values() for p in ps])
        total, _ = _total_norm_by_mesh(groups, error_if_nonfinite=False, foreach=True)
    if max_norm is not None and float(max_norm) > 0:
        _clip_grads_with_global_norm(groups, float(max_norm), total, foreach=True)
    return float(total)


def reduce_mean(values: list[float], device: str | torch.device = "cuda") -> float:
    """Mean of per-rank value lists across all ranks (count-weighted); nan when empty everywhere."""
    t = torch.tensor([float(sum(values)), float(len(values))], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t[0] / t[1]) if t[1] > 0 else float("nan")


def mean_or_nan(xs: list[float]) -> float:
    xs = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(sum(xs) / len(xs)) if xs else float("nan")


# --------------------------------------------------------------------------------------------
# theta
# --------------------------------------------------------------------------------------------
class ReptileMetaState:
    """theta = one local-shard copy of every trainable parameter; Reptile interpolation + restore.

    Args:
        net: ``model.net`` AFTER the optimizer was built (``requires_grad`` marks the trainable set).
        meta_optimizer: ``"sgd"`` (theta += eps * d, the original Reptile) or ``"adam"`` (Adam on ``d``).
    """

    def __init__(
        self,
        net: nn.Module,
        meta_optimizer: str = "sgd",
        meta_betas: tuple[float, float] = (0.9, 0.999),
        meta_eps: float = 1e-8,
    ) -> None:
        if meta_optimizer not in ("sgd", "adam"):
            raise ValueError(f"meta_optimizer must be 'sgd' or 'adam', got {meta_optimizer!r}")
        self.net = net
        self.params: dict[str, nn.Parameter] = {n: p for n, p in net_param_lookup(net).items() if p.requires_grad}
        if not self.params:
            raise ValueError("no trainable parameters -- build the optimizer (keys_to_select) before ReptileMetaState")
        self.groups: dict[str, str] = {n: param_group(n) for n in self.params}
        self.theta: dict[str, torch.Tensor] = {n: local_tensor(p.data).detach().clone() for n, p in self.params.items()}
        self.meta_optimizer = meta_optimizer
        self.meta_betas = (float(meta_betas[0]), float(meta_betas[1]))
        self.meta_eps = float(meta_eps)
        self.adam_m: dict[str, torch.Tensor] | None = None
        self.adam_v: dict[str, torch.Tensor] | None = None
        self.adam_t = 0
        self.accum: dict[str, torch.Tensor] | None = None
        self.accum_weight = 0.0

    # ---- bookkeeping ---------------------------------------------------------------------
    def numel_by_group(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n, p in self.params.items():
            out[self.groups[n]] = out.get(self.groups[n], 0) + int(p.numel())  # DTensor.numel() = global size
        return dict(sorted(out.items()))

    @property
    def numel(self) -> int:
        return int(sum(self.numel_by_group().values()))

    def _device(self) -> torch.device:
        return next(iter(self.theta.values())).device

    def _all_reduce_sum(self, values: dict[str, torch.Tensor]) -> dict[str, float]:
        if not values:
            return {}
        keys = sorted(values)
        t = torch.stack([values[k].float() for k in keys])
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return {k: float(v) for k, v in zip(keys, t.tolist(), strict=True)}

    @torch.no_grad()
    def displacement_norms_by_group(self) -> dict[str, float]:
        """``||theta_tilde - theta||_2`` per group (theta_tilde = the model's current weights). Collective."""
        sq: dict[str, torch.Tensor] = {}
        for n, p in self.params.items():
            d = local_tensor(p.data).float() - self.theta[n].float()
            g = self.groups[n]
            sq[g] = sq.get(g, torch.zeros((), device=d.device)) + (d * d).sum()
        return {g: v**0.5 for g, v in self._all_reduce_sum(sq).items()}

    @torch.no_grad()
    def theta_norms_by_group(self) -> dict[str, float]:
        sq: dict[str, torch.Tensor] = {}
        for n in self.params:
            t = self.theta[n].float()
            g = self.groups[n]
            sq[g] = sq.get(g, torch.zeros((), device=t.device)) + (t * t).sum()
        return {g: v**0.5 for g, v in self._all_reduce_sum(sq).items()}

    # ---- fast weights <-> theta -----------------------------------------------------------
    @torch.no_grad()
    def restore_to_model(self) -> None:
        """Write theta into the model (start of a task / after the meta step)."""
        for n, p in self.params.items():
            local_tensor(p.data).copy_(self.theta[n])

    @torch.no_grad()
    def accumulate_displacement(self, weight: float = 1.0) -> None:
        """Add ``weight * (theta_tilde - theta)`` of the finished task to the batch (``all_datasets`` mode)."""
        if self.accum is None:
            self.accum = {n: torch.zeros_like(self.theta[n]) for n in self.params}
        for n, p in self.params.items():
            self.accum[n].add_(local_tensor(p.data) - self.theta[n], alpha=float(weight))
        self.accum_weight += float(weight)

    @torch.no_grad()
    def meta_step(self, eps: float) -> float:
        """theta <- theta + eps * d with d the (weight-averaged) displacement; model <- theta.
        Returns the global L2 norm of the applied update."""
        use_accum = self.accum is not None and self.accum_weight > 0
        sq_update = torch.zeros((), device=self._device())
        if self.meta_optimizer == "adam":
            self.adam_t += 1
            b1, b2 = self.meta_betas
            if self.adam_m is None:
                self.adam_m = {n: torch.zeros_like(self.theta[n]) for n in self.params}
                self.adam_v = {n: torch.zeros_like(self.theta[n]) for n in self.params}
        for n, p in self.params.items():
            if use_accum:
                d = self.accum[n].div_(float(self.accum_weight))  # type: ignore[index]
            else:
                d = local_tensor(p.data) - self.theta[n]
            if self.meta_optimizer == "sgd":
                upd = d.mul_(float(eps))
            else:
                m, v = self.adam_m[n], self.adam_v[n]  # type: ignore[index]
                m.mul_(b1).add_(d, alpha=1 - b1)
                v.mul_(b2).addcmul_(d, d, value=1 - b2)
                m_hat = m / (1 - b1**self.adam_t)
                v_hat = v / (1 - b2**self.adam_t)
                upd = (m_hat / (v_hat.sqrt() + self.meta_eps)).mul_(float(eps))
            sq_update = sq_update + (upd.float() * upd.float()).sum()
            self.theta[n].add_(upd.to(self.theta[n].dtype))
        self.accum = None
        self.accum_weight = 0.0
        self.restore_to_model()
        return self._all_reduce_sum({"u": sq_update})["u"] ** 0.5

    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": "reptile",
            "num_params": len(self.params),
            "numel_by_group": self.numel_by_group(),
            "meta_optimizer": self.meta_optimizer,
        }


__all__ = [
    "GROUP_ACTION_HEADS",
    "ReptileMetaState",
    "canonical_param_name",
    "capture_base_lrs",
    "clip_grads",
    "local_tensor",
    "mean_or_nan",
    "net_param_lookup",
    "param_group",
    "reduce_mean",
    "reset_optimizer_state",
    "set_lr_scale",
]
