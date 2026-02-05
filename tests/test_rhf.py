from ad_afqmc_prototype import config

config.configure_once()

import jax.numpy as jnp
import pytest
from pyscf import gto, scf

from ad_afqmc_prototype import driver
from ad_afqmc_prototype.afqmc import AFQMC
from ad_afqmc_prototype.prop.types import QmcParams


def run_calc(
    sys, meas_ops, ham_data, trial_ops, trial_data, params, block_fn, prop_ops
):
    mean, err, block_e_all, block_w_all = driver.run_qmc_energy(
        sys=sys,
        params=params,
        ham_data=ham_data,
        trial_ops=trial_ops,
        trial_data=trial_data,
        meas_ops=meas_ops,
        prop_ops=prop_ops,
        block_fn=block_fn,
    )
    return mean, err, block_e_all, block_w_all


@pytest.mark.parametrize(
    "walker_kind, e_ref, err_ref",
    [
        ("restricted", -75.75594174131398, 0.01213379336719581),
        ("unrestricted", -75.75594174131398, 0.01213379336719581),
    ],
)
def test_calc_rhf_hamiltonian(mf, params, walker_kind, e_ref, err_ref):
    myafqmc = AFQMC(mf)
    myafqmc.params = params
    myafqmc.walker_kind = walker_kind
    myafqmc.mixed_precision = False
    myafqmc.chol_cut = 1e-6
    mean, err = myafqmc.kernel()
    assert jnp.isclose(mean, e_ref), (mean, e_ref, mean - e_ref)
    assert jnp.isclose(err, err_ref), (err, err_ref, err - err_ref)


@pytest.fixture(scope="module")
def mf():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
    )
    mf = scf.RHF(mol)
    mf.kernel()
    return mf


@pytest.fixture(scope="module")
def params():
    return QmcParams(
        n_eql_blocks=4,
        n_blocks=20,
        seed=1234,
        n_walkers=5,
    )


if __name__ == "__main__":
    pytest.main([__file__])
