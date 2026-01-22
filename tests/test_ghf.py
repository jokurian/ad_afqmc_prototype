from ad_afqmc_prototype import config

config.setup_jax()

import jax
import jax.numpy as jnp
import pytest

from ad_afqmc_prototype.core.ops import k_energy, k_force_bias
from ad_afqmc_prototype.core.system import System
from ad_afqmc_prototype.ham.chol import HamChol
from ad_afqmc_prototype.meas.auto import make_auto_meas_ops
from ad_afqmc_prototype.meas.ghf import make_ghf_meas_ops_chol
from ad_afqmc_prototype.trial.ghf import GhfTrial, make_ghf_trial_ops
from ad_afqmc_prototype import testing

def _make_ghf_trial(key, norb, nup, ndn, dtype=jnp.complex128) -> GhfTrial:
    ne = nup + ndn
    mo = testing._rand_orthonormal_cols(key, 2 * norb, ne, dtype=dtype)
    return GhfTrial(mo_coeff=mo)

@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_force_bias_matches_manual_ghf(walker_kind, norb, nup, ndn, n_chol):
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)

    key = jax.random.PRNGKey(0)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = testing._make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_ghf_trial(k_trial, norb=norb, nup=nup, ndn=ndn)

    t_ops = make_ghf_trial_ops(sys)
    meas_manual = make_ghf_meas_ops_chol(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    fb_manual = meas_manual.require_kernel(k_force_bias)
    fb_auto = meas_auto.require_kernel(k_force_bias)

    for i in range(4):
        wi = testing._make_walkers(jax.random.fold_in(k_w, i), sys)
        v_m = fb_manual(wi, ham, ctx_manual, trial)
        v_a = fb_auto(wi, ham, ctx_auto, trial)

        # AUTO uses overlap-derivatives; for GHF you may need slightly looser tol than RHF.
        assert jnp.allclose(v_a, v_m, rtol=5e-6, atol=5e-7), (v_a, v_m)


@pytest.mark.parametrize(
    "walker_kind,norb,nup,ndn,n_chol",
    [
        ("restricted", 6, 2, 2, 8),
        ("unrestricted", 6, 2, 1, 8),
        ("generalized", 6, 2, 1, 8),
    ],
)
def test_auto_energy_matches_manual_ghf(walker_kind, norb, nup, ndn, n_chol):
    sys = System(norb=norb, nelec=(nup, ndn), walker_kind=walker_kind)

    key = jax.random.PRNGKey(1)
    k_ham, k_trial, k_w = jax.random.split(key, 3)

    ham = testing._make_random_ham_chol(k_ham, norb=norb, n_chol=n_chol)
    trial = _make_ghf_trial(k_trial, norb=norb, nup=nup, ndn=ndn)

    t_ops = make_ghf_trial_ops(sys)
    meas_manual = make_ghf_meas_ops_chol(sys)
    meas_auto = make_auto_meas_ops(sys, t_ops, eps=1.0e-4)

    ctx_manual = meas_manual.build_meas_ctx(ham, trial)
    ctx_auto = meas_auto.build_meas_ctx(ham, trial)

    # Some implementations may not define energy for some walker kinds; skip in that case.
    if not meas_manual.has_kernel(k_energy):
        pytest.skip(
            f"manual GHF meas does not provide '{k_energy}' for walker_kind={walker_kind}"
        )

    e_manual = meas_manual.require_kernel(k_energy)
    e_auto = meas_auto.require_kernel(k_energy)

    for i in range(4):
        wi = testing._make_walkers(jax.random.fold_in(k_w, i), sys)
        em = e_manual(wi, ham, ctx_manual, trial)
        ea = e_auto(wi, ham, ctx_auto, trial)

        assert jnp.allclose(ea, em, rtol=5e-6, atol=5e-7), (ea, em)


if __name__ == "__main__":
    pytest.main([__file__])
