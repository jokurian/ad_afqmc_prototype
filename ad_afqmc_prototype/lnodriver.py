from __future__ import annotations

import time
from functools import partial
from pprint import pprint
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .core.ops import MeasOps, TrialOps
from .core.system import System
from .prop.blocks import BlockFn, BlockObs
from .prop.types import PropOps, PropState, QmcParams
from .stat_utils import blocking_analysis_ratio, reject_outliers
from .walkers import stochastic_reconfiguration

print = partial(print, flush=True)

def proc_fn_lno(samples: Array, state: PropState, params: QmcParams) -> BlockObs:
    (e_samples,ecorr_samples) = samples
    ecorr_samples = jnp.real(ecorr_samples)
    e_samples = jnp.real(e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    e_samples = jnp.where(jnp.abs(e_samples - e_ref) > thresh, e_ref, e_samples)

    weights = state.weights
    w_sum = jnp.sum(weights)
    w_sum_safe = jnp.where(w_sum == 0, 1.0, w_sum)
    e_block = jnp.sum(weights * e_samples) / w_sum_safe
    e_block = jnp.where(w_sum == 0, e_ref, e_block)

    obs = BlockObs(scalars={"energy": e_block, "weight": w_sum, "ecorr": jnp.sum(weights * ecorr_samples) / w_sum_safe})
    return obs


def make_run_blocks(
    *,
    block_fn: BlockFn,
    sys: System,
    params: QmcParams,
    trial_ops: TrialOps,
    meas_ops: MeasOps,
    prop_ops: PropOps,
) -> Callable:
    """
    Build a jitted run_blocks.
    We keep ham_data, trial_data, meas_ctx, prop_ctx as arguments to
    improve compilation, as these objects can be large.
    """

    @partial(jax.jit, static_argnames=("n_blocks",))
    def run_blocks(
        state0,
        *,
        ham_data,
        trial_data,
        meas_ctx,
        prop_ctx,
        n_blocks: int,
    ):
        def one_block(state, _):
            state, obs = block_fn(
                state,
                sys=sys,
                params=params,
                ham_data=ham_data,
                trial_data=trial_data,
                trial_ops=trial_ops,
                meas_ops=meas_ops,
                meas_ctx=meas_ctx,
                prop_ops=prop_ops,
                prop_ctx=prop_ctx,
                k_names= ("energy","lnoenergy"),
                proc_fn= proc_fn_lno
            )
            return state, (obs.scalars["energy"], obs.scalars["weight"],obs.scalars["ecorr"])

        stateN, (e, w, ecorr) = lax.scan(one_block, state0, xs=None, length=n_blocks)
        return stateN, e, w, ecorr

    return run_blocks


def run_lnoqmc_energy(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    trial_data: Any,
    meas_ops: MeasOps,
    trial_ops: TrialOps,
    prop_ops: PropOps,
    block_fn: BlockFn,
    state: PropState | None = None,
    meas_ctx: Any | None = None,
    target_error: float | None = None,
    mesh: Mesh | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    equilibration blocks then sampling blocks.

    Returns:
      (mean_energy, stderr, block_energies, block_weights, block_ecorr)
    """
    # build ctx
    prop_ctx = prop_ops.build_prop_ctx(ham_data, trial_ops.get_rdm1(trial_data), params)
    if meas_ctx is None:
        meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
    if state is None:
        state = prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=trial_ops,
            trial_data=trial_data,
            meas_ops=meas_ops,
            params=params,
            mesh=mesh,
        )

    if mesh is None or mesh.size == 1:
        block_fn_sr = block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        sr_sharded = partial(stochastic_reconfiguration, data_sharding=data_sh)
        block_fn_sr = partial(block_fn, sr_fn=sr_sharded)

    run_blocks = make_run_blocks(
        block_fn=block_fn_sr,
        sys=sys,
        params=params,
        trial_ops=trial_ops,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
    )

    t0 = time.perf_counter()
    t_mark = t0

    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    block_e_eq = []
    block_ecorr_eq = []
    block_w_eq = []
    block_e_eq.append(state.e_estimate)
    block_ecorr_eq.append(0.0)
    block_w_eq.append(jnp.sum(state.weights))
    print("\nEquilibration:\n")
    if print_every:
        print(
            f"{'':4s}"
            f"{'block':>9s}  "
            f"{'E_blk':>14s}  "
            f"{'W':>12s}   "
            f"{'nodes':>10s}  "
            f"{'t[s]':>8s}"
        )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  "
        f"{float(state.e_estimate):14.10f}  "
        f"{float(jnp.sum(state.weights)):12.6e}  "
        f"{int(state.node_encounters):10d}  "
        f"{0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, e_chunk, w_chunk, ecorr_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_eq.extend(e_chunk.tolist())
        block_ecorr_eq.extend(ecorr_chunk.tolist())
        block_w_eq.extend(w_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        ecorr_chunk_avg = jnp.mean(ecorr_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  "
            f"{float(e_chunk_avg):14.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{float(ecorr_chunk_avg):12.6e}  "
            f"{int(state.node_encounters):10d}  "
            f"{elapsed:8.1f}"
        )
    block_e_eq = jnp.asarray(block_e_eq)
    block_w_eq = jnp.asarray(block_w_eq)
    block_ecorr_eq = jnp.asarray(block_ecorr_eq)

    # sampling
    print("\nSampling:\n")
    if target_error is None:
        target_error = 0.0
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    block_e_s = []
    block_w_s = []
    block_ecorr_s = []
    if print_every:
        print(
            f"{'':4s}{'block':>9s}  {'E_avg':>14s}  {'E_err':>10s}  {'E_block':>14s}  "
            f"{'W':>12s}    {'E_corr':>12s}    {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>7s}"
        )

    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, e_chunk, w_chunk, ecorr_chunk = run_blocks(
            state,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            n_blocks=n,
        )
        block_e_s.extend(e_chunk.tolist())
        block_w_s.extend(w_chunk.tolist())
        block_ecorr_s.extend(ecorr_chunk.tolist())
        w_chunk_avg = jnp.mean(w_chunk)
        e_chunk_avg = jnp.mean(e_chunk * w_chunk) / w_chunk_avg
        ecorr_chunk_avg = jnp.mean(ecorr_chunk * w_chunk) / w_chunk_avg
        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()
        stats = blocking_analysis_ratio(
            jnp.asarray(block_e_s), jnp.asarray(block_w_s), print_q=False
        )
        stats_ecorr = blocking_analysis_ratio(
            jnp.asarray(block_ecorr_s), jnp.asarray(block_w_s), print_q=False
        )
        mu = stats["mu"]
        se = stats["se_star"]
        mu_ecorr = stats_ecorr["mu"]
        se_ecorr = stats_ecorr["se_star"]
        nodes = int(state.node_encounters)
        print(
            f"[blk {start + n:4d}/{params.n_blocks}]  "
            f"{mu:14.10f}  "
            f"{(f'{se:10.3e}' if se is not None else ' ' * 10)}  "
            f"{float(e_chunk_avg):16.10f}  "
            f"{float(w_chunk_avg):12.6e}  "
            f"{float(mu_ecorr):12.6e}  "
            f"{nodes:10d}  "
            f"{dt_per_block:9.3f}  "
            f"{elapsed:8.1f}"
        )
        if se_ecorr is not None and se_ecorr <= target_error and target_error > 0.0:
            print(f"\nTarget error for orbital contribution {target_error:.3e} reached at block {start + n}.")
            break
    block_e_s = jnp.asarray(block_e_s)
    block_w_s = jnp.asarray(block_w_s)
    block_ecorr_s = jnp.asarray(block_ecorr_s)
    data_clean, _ = reject_outliers(jnp.column_stack((block_e_s, block_w_s)), obs=0)
    ecorr_data_clean, _ = reject_outliers(jnp.column_stack((block_ecorr_s, block_w_s)), obs=0)
    print(f"\nRejected {block_e_s.shape[0] - data_clean.shape[0]} outlier blocks.")
    block_e_s = jnp.asarray(data_clean[:, 0])
    block_w_s = jnp.asarray(data_clean[:, 1])
    block_ecorr_s = jnp.asarray(ecorr_data_clean[:, 0])
    block_ecorr_w_s = jnp.asarray(ecorr_data_clean[:, 1])
    print("\nFinal blocking analysis:")
    stats = blocking_analysis_ratio(block_e_s, block_w_s, print_q=True)
    stats_ecorr = blocking_analysis_ratio(block_ecorr_s, block_ecorr_w_s, print_q=True)
    mean, err = stats["mu"], stats["se_star"]
    mean_ecorr, err_ecorr = stats_ecorr["mu"], stats_ecorr["se_star"]

    block_e_all = jnp.concatenate([block_e_eq, block_e_s])
    block_w_all = jnp.concatenate([block_w_eq, block_w_s])
    block_ecorr_all = jnp.concatenate([block_ecorr_eq, block_ecorr_s])

    return mean, err, block_e_all, block_w_all, mean_ecorr, err_ecorr, block_ecorr_all
