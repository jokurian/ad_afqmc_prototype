from __future__ import annotations

from ..core.ops import overlap_fn, rdm1_fn, trial_ops
from ..core.system import system


def make_auto_trial_ops(
    sys: system,
    *,
    overlap_r: overlap_fn,
    overlap_u: overlap_fn,
    overlap_g: overlap_fn,
    get_rdm1: rdm1_fn,
) -> trial_ops:
    """
    For convenience.
    """
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        return trial_ops(overlap=overlap_r, get_rdm1=get_rdm1)

    if wk == "unrestricted":
        return trial_ops(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        return trial_ops(overlap=overlap_g, get_rdm1=get_rdm1)

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
