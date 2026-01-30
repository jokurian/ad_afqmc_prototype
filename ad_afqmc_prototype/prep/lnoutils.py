import numpy as np
from pyscf import __config__, ao2mo, mcscf, scf, lo
from ad_afqmc_prototype.prep.integrals import modified_cholesky
import os
import re
import jax.numpy as jnp


def run_afqmc_lno_mf(
    mf,
    integrals=None,
    norb_act=None,
    nelec_act=None,
    mo_coeff=None,
    norb_frozen=[],
    nproc=None,
    chol_cut=1e-5,
    seed=None,
    dt=0.005,
    nwalk_per_proc=5,
    nblocks=1000,
    maxError=1e-4,
    prjlo=None,
    tmpdir="./",
    output_file_name="afqmc_output.out",
    n_eql=2,
):
    import os
    os.makedirs(tmpdir, exist_ok=True)
    output_file_name = os.path.join(tmpdir, output_file_name)
    # print("#\n# Preparing AFQMC calculation")
    mol = mf.mol
    # choose the orbital basis
    if mo_coeff is None:
        if isinstance(mf, scf.uhf.UHF):
            mo_coeff = mf.mo_coeff[0]
        elif isinstance(mf, scf.rhf.RHF):
            mo_coeff = mf.mo_coeff
        else:
            raise Exception("# Invalid mean field object!")

    # calculate cholesky integrals
    # print("# Calculating Cholesky integrals")
    h1e, chol, nelec, enuc, nbasis, nchol = [None] * 6
    DFbas = mf.with_df.auxmol.basis
    nbasis = mol.nao

    mc = mcscf.CASSCF(mf, norb_act, nelec_act)
    mc.frozen = norb_frozen
    nelec = mc.nelecas
    mc.mo_coeff = mo_coeff
    h1e, enuc = mc.get_h1eff()
    nbasis = mo_coeff.shape[-1]
    act = [i for i in range(nbasis) if i not in norb_frozen]
    e = ao2mo.kernel(mf.mol, mo_coeff[:, act])#, compact=False)
    chol = modified_cholesky(e, max_error=chol_cut)

    nbasis = h1e.shape[-1]
    chol = chol.reshape((-1, nbasis, nbasis))
    v0 = 0.5 * np.einsum("nik,njk->ij", chol, chol, optimize="optimal")
    h1e_mod = h1e - v0
    # chol = chol.reshape((chol.shape[0], -1))

    # write mo coefficients
    trial_coeffs = np.empty((2, nbasis, nbasis))

    q = np.eye(mol.nao - len(norb_frozen))
    trial_coeffs[0] = q
    trial_coeffs[1] = q
    mo_coeff = trial_coeffs
    from ad_afqmc_prototype.wrapper.lnorhf import LnoRhf
    myafqmc = LnoRhf(mf)
    myafqmc.h0 = enuc
    myafqmc.h1 = jnp.array(h1e)
    myafqmc.chol = jnp.array(chol)
    myafqmc.mo_coeff = jnp.array(mo_coeff)[0][:,:nelec[0]]
    myafqmc.n_eql_blocks = n_eql
    myafqmc.n_blocks = nblocks
    myafqmc.dt = dt
    myafqmc.seed = seed if seed is not None else None
    myafqmc.n_walkers = nwalk_per_proc
    myafqmc.prjlo = jnp.array(prjlo) if prjlo is not None else None
    myafqmc.target_error = maxError
    # print("# Running AFQMC calculation")
    mean, err, block_e_all, block_w_all, mean_ecorr, err_ecorr, block_ecorr_all  = myafqmc.kernel()
    return mean_ecorr, err_ecorr 
    



def prep_local_orbitals(mf, frozen=0, localization_method="pm"):
    if localization_method not in ["pm"]:
        raise ValueError(
            f"Localization method '{localization_method}' is not supported. Make LOs by yourself."
        )
    orbocc = mf.mo_coeff[:, frozen : np.count_nonzero(mf.mo_occ)]
    mlo = lo.PipekMezey(mf.mol, orbocc)
    lo_coeff = mlo.kernel()
    while (
        True
    ):  # always performing jacobi sweep to avoid trapping in local minimum/saddle point
        lo_coeff1 = mlo.stability_jacobi()[1]
        if lo_coeff1 is lo_coeff:
            break
        mlo = lo.PipekMezey(mf.mol, lo_coeff1).set(verbose=4)
        mlo.init_guess = None
        lo_coeff = mlo.kernel()

    # Fragment list: for PM, every orbital corresponds to a fragment
    frag_lolist = [[i] for i in range(lo_coeff.shape[1])]

    return lo_coeff, frag_lolist
