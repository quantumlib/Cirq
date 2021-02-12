# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Estimation of fidelity associated with experimental circuit executions."""
from abc import abstractmethod
from dataclasses import dataclass
from typing import (
    List,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

import numpy as np
import pandas as pd
import scipy.optimize
import sympy

from cirq import ops
from cirq.circuits import Circuit
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits

if TYPE_CHECKING:
    import cirq
    import multiprocessing

THETA_SYMBOL, ZETA_SYMBOL, CHI_SYMBOL, GAMMA_SYMBOL, PHI_SYMBOL = sympy.symbols(
    'theta zeta chi gamma phi'
)
SQRT_ISWAP = ops.ISWAP ** 0.5


def benchmark_2q_xeb_fidelities(
    sampled_df: pd.DataFrame,
    circuits: Sequence['cirq.Circuit'],
    cycle_depths: Sequence[int],
    param_resolver: 'cirq.ParamResolverOrSimilarType' = None,
    pool: Optional['multiprocessing.pool.Pool'] = None,
):
    """Simulate and benchmark two-qubit XEB circuits.

    This uses the estimator from
    `cirq.experiments.fidelity_estimation.least_squares_xeb_fidelity_from_expectations`, but
    adapted for use on pandas DataFrames for efficient vectorized operation.

    Args:
         sampled_df: The sampled results to benchmark. This is likely produced by a call to
            `sample_2q_xeb_circuits`.
        circuits: The library of circuits corresponding to the sampled results in `sampled_df`.
        cycle_depths: The sequence of cycle depths to simulate the circuits.
        param_resolver: If circuits contain parameters, resolve according to this ParamResolver
            prior to simulation
        pool: If provided, execute the simulations in parallel.

    Returns:
        A DataFrame with columns 'cycle_depth' and 'fidelity'.
    """
    simulated_df = simulate_2q_xeb_circuits(
        circuits=circuits, cycle_depths=cycle_depths, param_resolver=param_resolver, pool=pool
    )
    df = sampled_df.join(simulated_df)

    def _summary_stats(row):
        D = 4  # Two qubits
        row['e_u'] = np.sum(row['pure_probs'] ** 2)
        row['u_u'] = np.sum(row['pure_probs']) / D
        row['m_u'] = np.sum(row['pure_probs'] * row['sampled_probs'])

        row['y'] = row['m_u'] - row['u_u']
        row['x'] = row['e_u'] - row['u_u']

        row['numerator'] = row['x'] * row['y']
        row['denominator'] = row['x'] ** 2
        return row

    df = df.apply(_summary_stats, axis=1)

    def per_cycle_depth(df):
        """This function is applied per cycle_depth in the following groupby aggregation."""
        fid_lsq = df['numerator'].sum() / df['denominator'].sum()
        ret = {'fidelity': fid_lsq}

        def _try_keep(k):
            """If all the values for a key `k` are the same in this group, we can keep it."""
            if k not in df.columns:
                return  # coverage: ignore
            vals = df[k].unique()
            if len(vals) == 1:
                ret[k] = vals[0]
            else:
                # coverage: ignore
                raise AssertionError(
                    f"When computing per-cycle-depth fidelity, multiple "
                    f"values for {k} were grouped together: {vals}"
                )

        _try_keep('q0')
        _try_keep('q1')
        _try_keep('pair_name')
        return pd.Series(ret)

    if 'pair_i' in df.columns:
        groupby_names = ['layer_i', 'pair_i', 'cycle_depth']
    else:
        groupby_names = ['cycle_depth']

    return df.reset_index().groupby(groupby_names).apply(per_cycle_depth).reset_index()


# mypy issue: https://github.com/python/mypy/issues/5374
@dataclass(frozen=True)  # type: ignore
class XEBPhasedFSimCharacterizationOptions:
    """Options for calibrating a PhasedFSim-like gate using XEB.

    You may want to use more specific subclasses like `SqrtISwapXEBOptions`
    which have sensible defaults.

    Attributes:
        characterize_theta: Whether to characterize θ angle.
        characterize_zeta: Whether to characterize ζ angle.
        characterize_chi: Whether to characterize χ angle.
        characterize_gamma: Whether to characterize γ angle.
        characterize_phi: Whether to characterize φ angle.
        theta_default: The initial or default value to assume for the θ angle.
        zeta_default: The initial or default value to assume for the ζ angle.
        chi_default: The initial or default value to assume for the χ angle.
        gamma_default: The initial or default value to assume for the γ angle.
        phi_default: The initial or default value to assume for the φ angle.
    """

    characterize_theta: bool = True
    characterize_zeta: bool = True
    characterize_chi: bool = True
    characterize_gamma: bool = True
    characterize_phi: bool = True

    theta_default: float = 0
    zeta_default: float = 0
    chi_default: float = 0
    gamma_default: float = 0
    phi_default: float = 0

    @staticmethod
    @abstractmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        """Whether to replace `op` with a parameterized version."""

    def get_initial_simplex_and_names(
        self, initial_simplex_step_size: float = 0.1
    ) -> Tuple[np.ndarray, List[str]]:
        """Get an initial simplex and parameter names for the optimization implied by these options.

        The initial simplex initiates the Nelder-Mead optimization parameter. We
        use the standard simplex of `x0 + s*basis_vec` where x0 is given by the
        `xxx_default` attributes, s is `initial_simplex_step_size` and `basis_vec`
        is a one-hot encoded vector for each parameter for which the `parameterize_xxx`
        attribute is True.

        We also return a list of parameter names so the Cirq `param_resovler`
        can be accurately constructed during optimization.
        """
        x0 = []
        names = []
        if self.characterize_theta:
            x0 += [self.theta_default]
            names += [THETA_SYMBOL.name]
        if self.characterize_zeta:
            x0 += [self.zeta_default]
            names += [ZETA_SYMBOL.name]
        if self.characterize_chi:
            x0 += [self.chi_default]
            names += [CHI_SYMBOL.name]
        if self.characterize_gamma:
            x0 += [self.gamma_default]
            names += [GAMMA_SYMBOL.name]
        if self.characterize_phi:
            x0 += [self.phi_default]
            names += [PHI_SYMBOL.name]

        x0 = np.asarray(x0)
        n_param = len(x0)
        initial_simplex = [x0]
        for i in range(n_param):
            basis_vec = np.eye(1, n_param, i)[0]
            initial_simplex += [x0 + initial_simplex_step_size * basis_vec]
        initial_simplex = np.asarray(initial_simplex)

        return initial_simplex, names


@dataclass(frozen=True)
class SqrtISwapXEBOptions(XEBPhasedFSimCharacterizationOptions):
    """Options for calibrating a sqrt(ISWAP) gate using XEB.

    As such, the default for theta is changed to -pi/4 and the parameterization
    predicate seeks out sqrt(ISWAP) gates.
    """

    theta_default: float = -np.pi / 4

    @staticmethod
    def should_parameterize(op: 'cirq.Operation') -> bool:
        return op.gate == SQRT_ISWAP


def parameterize_phased_fsim_circuit(
    circuit: 'cirq.Circuit',
    phased_fsim_options: XEBPhasedFSimCharacterizationOptions,
) -> 'cirq.Circuit':
    """Parameterize PhasedFSim-like gates in a given circuit according to
    `phased_fsim_options`.
    """
    options = phased_fsim_options
    theta = THETA_SYMBOL if options.characterize_theta else options.theta_default
    zeta = ZETA_SYMBOL if options.characterize_zeta else options.zeta_default
    chi = CHI_SYMBOL if options.characterize_chi else options.chi_default
    gamma = GAMMA_SYMBOL if options.characterize_gamma else options.gamma_default
    phi = PHI_SYMBOL if options.characterize_phi else options.phi_default

    fsim_gate = ops.PhasedFSimGate(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi)
    return Circuit(
        ops.Moment(
            fsim_gate.on(*op.qubits) if options.should_parameterize(op) else op
            for op in moment.operations
        )
        for moment in circuit.moments
    )


def characterize_phased_fsim_parameters_with_xeb(
    sampled_df: pd.DataFrame,
    parameterized_circuits: List['cirq.Circuit'],
    cycle_depths: Sequence[int],
    phased_fsim_options: XEBPhasedFSimCharacterizationOptions,
    initial_simplex_step_size: float = 0.1,
    xatol: float = 1e-3,
    fatol: float = 1e-3,
    verbose: bool = True,
    pool: Optional['multiprocessing.pool.Pool'] = None,
):
    """Run a classical optimization to fit phased fsim parameters to experimental data, and
    thereby characterize PhasedFSim-like gates.

    Args:
        sampled_df: The DataFrame of sampled two-qubit probability distributions returned
            from `sample_2q_xeb_circuits`.
        parameterized_circuits: The circuits corresponding to those sampled in `sampled_df`,
            but with some gates parameterized, likely by using `parameterize_phased_fsim_circuit`.
        cycle_depths: The depths at which circuits were truncated.
        phased_fsim_options: A set of options that controls the classical optimization loop
            for characterizing the parameterized gates.
        initial_simplex_step_size: Set the size of the initial simplex for Nelder-Mead.
        xatol: The `xatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the parameters.
        fatol: The `fatol` argument for Nelder-Mead. This is the absolute error for convergence
            in the function evaluation.
        verbose: Whether to print progress updates.
        pool: An optional multiprocessing pool to execute circuit simulations in parallel.
    """
    initial_simplex, names = phased_fsim_options.get_initial_simplex_and_names(
        initial_simplex_step_size=initial_simplex_step_size
    )
    x0 = initial_simplex[0]

    def _mean_infidelity(angles):
        params = dict(zip(names, angles))
        if verbose:
            params_str = ''
            for name, val in params.items():
                params_str += f'{name:5s} = {val:7.3g} '
            print("Simulating with {}".format(params_str))
        fids = benchmark_2q_xeb_fidelities(
            sampled_df, parameterized_circuits, cycle_depths, param_resolver=params, pool=pool
        )

        loss = 1 - fids['fidelity'].mean()
        if verbose:
            print("Loss: {:7.3g}".format(loss), flush=True)
        return loss

    res = scipy.optimize.minimize(
        _mean_infidelity,
        x0=x0,
        options={'initial_simplex': initial_simplex, 'xatol': xatol, 'fatol': fatol},
        method='nelder-mead',
    )
    return res
