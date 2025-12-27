from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, NamedTuple, Protocol

import jax

from .typing import ham_data, trial_data

# Using Protocols for public APIs, Callables for internal helper APIs.

# trial


class overlap_fn(Protocol):
    def __call__(self, walker: Any, trial_data: Any) -> jax.Array: ...


class rdm1_fn(Protocol):
    def __call__(self, trial_data: Any) -> jax.Array: ...


class trial_ops(NamedTuple):
    """
    Trial operations.
      - overlap: overlap for a single walker
      - get_rdm1: trial rdm1
    """

    overlap: overlap_fn  # (walker, trial_data) -> overlap
    get_rdm1: rdm1_fn  # (trial_data) -> rdm1


# hamiltonian


class ham_ops(NamedTuple):
    """
    Hamiltonian (would probably be helpful when adding different Hamiltonians).
    """

    n_fields: Callable[[ham_data], int]


# measurements


class meas_kernel(Protocol):
    """
    Measurement kernel protocol.
    """

    def __call__(
        self, walker: Any, ham_data: Any, meas_ctx: Any, trial_data: Any
    ) -> jax.Array: ...


# usual kernel names
k_energy = "energy"
k_force_bias = "force_bias"


@dataclass(frozen=True)
class meas_ops:
    """
    Measurement ops: trial + ham estimators + optional observables.
    """

    # same as trial_ops.overlap
    overlap: overlap_fn  # (walker, trial_data) -> overlap

    # intermediates for measurements
    build_meas_ctx: Callable[[ham_data, trial_data], Any] = (
        lambda ham_data, trial_data: None
    )

    # algorithm kernels (e.g. "energy", "force_bias")
    kernels: Mapping[str, meas_kernel] = field(default_factory=dict)

    # optional observables (e.g. "mixed_rdm1", "nn_corr", ...)
    observables: Mapping[str, meas_kernel] = field(default_factory=dict)

    def has_kernel(self, name: str) -> bool:
        return name in self.kernels

    def has_observable(self, name: str) -> bool:
        return name in self.observables

    def require_kernel(self, name: str) -> meas_kernel:
        try:
            return self.kernels[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.kernels.keys()))
            raise KeyError(
                f"missing required kernel '{name}'. available: [{avail}]"
            ) from e

    def require_observable(self, name: str) -> meas_kernel:
        try:
            return self.observables[name]
        except KeyError as e:
            avail = ", ".join(sorted(self.observables.keys()))
            raise KeyError(
                f"missing requested observable '{name}'. available: [{avail}]"
            ) from e

    def available_kernels(self) -> tuple[str, ...]:
        return tuple(sorted(self.kernels.keys()))

    def available_observables(self) -> tuple[str, ...]:
        return tuple(sorted(self.observables.keys()))
