import numpy as np
import jax.numpy as jnp
from functools import partial
from .. import config, lnodriver
from ..core.system import System
from ..ham.chol import HamChol
from ..meas.rhf import overlap_r,make_build_lno_meas_ctx,force_bias_kernel_rw_rh, energy_kernel_rw_rh, lnoenergy_kernel_rw_rh
from ..prop.afqmc import make_prop_ops
from ..prop.blocks import block
from ..prop.types import QmcParams
from ..trial.rhf import RhfTrial, make_rhf_trial_ops

from ..core.ops import MeasOps, k_energy, k_force_bias, k_lnoenergy
from ..meas.rhf import make_build_lno_meas_ctx

import numpy as np
import jax.numpy as jnp

class LNOAFQMC: #Right now only works for RHF trial and walkers
    def __init__(
        self,
        mf,
        prjlo=None,
        dt=0.005,
        seed=None,
        n_eql_blocks=50,
        n_blocks=500,
        mo_coeff=None,
        h0=None,
        h1=None,
        chol=None,
        n_chunks=1,
        n_exp_terms=6,
        pop_control_damping=0.1,
        weight_floor=1.0e-3,
        weight_cap=100.0,
        n_prop_steps=50,
        shift_ema=0.1,
        n_walkers=200,
        target_error=0.001,
        use_gpu=False,
        single_precision=False,
        quiet=False,
    ):
        self.quiet = quiet
        self.use_gpu = use_gpu
        self.single_precision = single_precision
        config.setup_jax(use_gpu=self.use_gpu, single_precision=self.single_precision, quiet=self.quiet)

        self.mf = mf
        self.mol = mf.mol

        # store scalar params
        self.n_eql_blocks = n_eql_blocks
        self.n_blocks = n_blocks
        self.seed = seed if seed is not None else np.random.randint(0, int(1e6))
        self.dt = dt
        self.n_chunks = n_chunks
        self.n_exp_terms = n_exp_terms
        self.pop_control_damping = pop_control_damping
        self.weight_floor = weight_floor
        self.weight_cap = weight_cap
        self.n_prop_steps = n_prop_steps
        self.shift_ema = shift_ema
        self.n_walkers = n_walkers

        self.prjlo = None if prjlo is None else jnp.asarray(prjlo)
        self.h0 = None if h0 is None else jnp.asarray(h0)
        self.h1 = None if h1 is None else jnp.asarray(h1)
        self.chol = None if chol is None else jnp.asarray(chol)
        self.mo_coeff = None if mo_coeff is None else jnp.asarray(mo_coeff)
        self.target_error = target_error  
          

        self._built = False

    def setup(self, prjlo, mo_coeff, h0, h1, chol):
        """Build everything that depends on prjlo/mo_coeff/integrals."""
        if prjlo is None:
            raise ValueError("setup() requires prjlo (array-like).")
        if mo_coeff is None:
            raise ValueError("setup() requires mo_coeff (array-like).")
        if h0 is None or h1 is None or chol is None:
            raise ValueError("setup() requires h0, h1, chol (array-like).")

        self.prjlo = jnp.asarray(prjlo)
        self.mo_coeff = jnp.asarray(mo_coeff)
        self.h0 = jnp.asarray(h0)
        self.h1 = jnp.asarray(h1)
        self.chol = jnp.asarray(chol)
        norb = mo_coeff.shape[0]
        nelec = mo_coeff.shape[1]
        mol = self.mol

        self.sys = System(norb=norb, nelec=[nelec, nelec], walker_kind="restricted")
        self.ham_data = HamChol(self.h0, self.h1, self.chol)
        self.trial_data = RhfTrial(self.mo_coeff[:, : nelec])
        self.trial_ops = make_rhf_trial_ops(sys=self.sys)
        self.meas_ops = MeasOps(
            overlap=overlap_r,
            build_meas_ctx= make_build_lno_meas_ctx(self.prjlo),
            kernels={
                k_force_bias: force_bias_kernel_rw_rh,
                k_energy: energy_kernel_rw_rh,
                k_lnoenergy: lnoenergy_kernel_rw_rh,
            },
        )


        self.prop_ops = make_prop_ops(ham_basis="restricted", walker_kind=self.sys.walker_kind)

        self.params = QmcParams(
            n_eql_blocks=self.n_eql_blocks,
            n_blocks=self.n_blocks,
            seed=np.random.randint(0, int(1e6)),
            n_walkers=self.n_walkers,
            dt=self.dt,
            n_chunks=self.n_chunks,
            n_exp_terms=self.n_exp_terms,
            pop_control_damping=self.pop_control_damping,
            weight_floor=self.weight_floor,
            weight_cap=self.weight_cap,
            n_prop_steps=self.n_prop_steps,
            shift_ema=self.shift_ema,
        )
        self.initial_walkers = None 
        self.block_fn = block
        return self


    def kernel(self):
        self.setup(self.prjlo, self.mo_coeff, self.h0, self.h1, self.chol)
        return lnodriver.run_lnoqmc_energy(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            trial_ops=self.trial_ops,
            trial_data=self.trial_data,
            meas_ops=self.meas_ops,
            prop_ops=self.prop_ops,
            block_fn=self.block_fn,
            target_error=self.target_error,
        )
