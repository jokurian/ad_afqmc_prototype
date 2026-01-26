from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdTrial:
    """
    Unrestricted CISD trial in an MO basis where the reference
    determinant occupies the first nocc[0] alpha and nocc[1] beta orbitals.

    Arrays:
      mo_coeff_b: (norb, nocc[1])
      c1a : (nocc[0], nvir[0])                      singles coefficients c_{i,alpha a,alpha}
      c1b : (nocc[1], nvir[1])                      singles coefficients c_{i,beta  a,beta }
      c2aa: (nocc[0], nvir[0], nocc[0], nvir[0])    doubles coefficients c_{i,alpha a,alpha j,alpha b,alpha}
      c2ab: (nocc[0], nvir[0], nocc[1], nvir[1])    doubles coefficients c_{i,alpha a,alpha j,beta  b,beta }
      c2bb: (nocc[1], nvir[1], nocc[1], nvir[1])    doubles coefficients c_{i,beta  a,beta  j,beta  b,beta }
    """
    mo_coeff_a: jax.Array
    mo_coeff_b: jax.Array
    c1a: jax.Array
    c1b: jax.Array
    c2aa: jax.Array
    c2ab: jax.Array
    c2bb: jax.Array

    @property
    def norb(self) -> int:
        return int(self.mo_coeff_b.shape[0])

    @property
    def nocc(self) -> tuple[int, int]:
        return (
            int(self.c1a.shape[0]),
            int(self.c1b.shape[0])
        )

    @property
    def nvir(self) -> tuple[int, int]:
        return (
            int(self.c1a.shape[1]),
            int(self.c1b.shape[1])
        )

    def tree_flatten(self):
        return (
            self.mo_coeff_a,
            self.mo_coeff_b,
            self.c1a,
            self.c1b,
            self.c2aa,
            self.c2ab,
            self.c2bb,
        ), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (
            mo_coeff_a,
            mo_coeff_b,
            c1a,
            c1b,
            c2aa,
            c2ab,
            c2bb,
        ) = children
        return cls(
            mo_coeff_a=mo_coeff_a,
            mo_coeff_b=mo_coeff_b, 
            c1a=c1a,
            c1b=c1b,
            c2aa=c2aa,
            c2ab=c2ab,
            c2bb=c2bb,
        )

def _det(m: jax.Array) -> jax.Array:
    return jnp.linalg.det(m)

def get_rdm1(trial_data: UcisdTrial) -> jax.Array:
    # UHF
    norb, (n_oa, n_ob) = trial_data.norb, trial_data.nocc
    occ_a = jnp.arange(norb) < n_oa
    c_b = trial_data.mo_coeff_b
    dm_a = jnp.diag(occ_a)     # (norb, norb)
    dm_b = c_b @ c_b.conj().T  # (norb, norb)
    return jnp.stack([dm_a, dm_b], axis=0)  # (2, norb, norb)


def overlap_u(walker: tuple[jax.Array, jax.Array], trial_data: UcisdTrial) -> jax.Array:
    wa, wb = walker
    n_oa, n_ob = trial_data.nocc
    c1a = trial_data.c1a
    c1b = trial_data.c1b
    c2aa = trial_data.c2aa
    c2ab = trial_data.c2ab
    c2bb = trial_data.c2bb
    c_b = trial_data.mo_coeff_b

    wb = c_b.T.dot(wb[:, :n_ob])
    woa = wa[:n_oa, :]  # (n_oa, n_oa)
    wob = wb[:n_ob, :]  # (n_ob, n_ob)

    g_a = jnp.linalg.solve(woa.T, wa.T)  # (n_oa, norb)
    g_b = jnp.linalg.solve(wob.T, wb.T)  # (n_ob, norb)

    g_a = g_a[:, n_oa:]
    g_b = g_b[:, n_ob:]
    o0 = jnp.linalg.det(woa) * jnp.linalg.det(wob)
    o1 = jnp.einsum("ia,ia", c1a, g_a) + jnp.einsum("ia,ia", c1b, g_b)
    o2 = (
        0.5 * jnp.einsum("iajb, ia, jb", c2aa, g_a, g_a)
        + 0.5 * jnp.einsum("iajb, ia, jb", c2bb, g_b, g_b)
        + jnp.einsum("iajb, ia, jb", c2ab, g_a, g_b)
    )
    return (1.0 + o1 + o2) * o0


def make_ucisd_trial_ops(sys: System) -> TrialOps:
    wk = sys.walker_kind.lower()

    if wk == "restricted":
        raise NotImplementedError

    if wk == "unrestricted":
        return TrialOps(overlap=overlap_u, get_rdm1=get_rdm1)

    if wk == "generalized":
        raise NotImplementedError

    raise ValueError(f"unknown walker_kind: {sys.walker_kind}")
