# Copyright 2024 The Cirq Developers
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

"""Provides a method to do z-phase calibration for excitation-preserving gates."""
import multiprocessing
import multiprocessing.pool
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union

import matplotlib.pyplot as plt
import numpy as np

from cirq import circuits, ops, protocols
from cirq.experiments import xeb_fitting
from cirq.experiments.two_qubit_xeb import parallel_xeb_workflow
from cirq.transformers import transformer_api

if TYPE_CHECKING:
    import pandas as pd

    import cirq


def z_phase_calibration_workflow(
    sampler: 'cirq.Sampler',
    qubits: Optional[Sequence['cirq.GridQubit']] = None,
    two_qubit_gate: 'cirq.Gate' = ops.CZ,
    options: Optional[xeb_fitting.XEBPhasedFSimCharacterizationOptions] = None,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    atol: float = 1e-3,
    num_workers_or_pool: Union[int, 'multiprocessing.pool.Pool'] = -1,
    pairs: Optional[Sequence[Tuple['cirq.GridQubit', 'cirq.GridQubit']]] = None,
    tags: Sequence[Any] = (),
) -> Tuple[xeb_fitting.XEBCharacterizationResult, 'pd.DataFrame']:
    """Perform z-phase calibration for excitation-preserving gates.

    For a given excitation-preserving two-qubit gate we assume an error model that can be described
    using Z-rotations:
                0: ───Rz(a)───two_qubit_gate───Rz(c)───
                                │
                1: ───Rz(b)───two_qubit_gate───Rz(d)───
    for some angles a, b, c, and d.

    Since the two-qubit gate is a excitation-preserving-gate, it can be represented by an FSimGate
    and the effect of rotations turns it into a PhasedFSimGate. Using XEB-data we find the
    PhasedFSimGate parameters that minimize the infidelity of the gate.

    References:
        - https://arxiv.org/abs/2001.08343
        - https://arxiv.org/abs/2010.07965
        - https://arxiv.org/abs/1910.11333

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Qubits to use. If none, use all qubits on the sampler's device.
        two_qubit_gate: The entangling gate to use.
        options: The XEB-fitting options. If None, calibrate only the three phase angles
            (chi, gamma, zeta) using the representation of a two-qubit gate as an FSimGate
            for the initial guess.
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
        random_state: The random state to use.
        atol: Absolute tolerance to be used by the minimizer.
        num_workers_or_pool: An optional multi-processing pool or number of workers.
            A zero value means no multiprocessing.
            A positive integer value will create a pool with the given number of workers.
            A negative value will create pool with maximum number of workers.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        tags: Tags to add to two qubit operations.
    Returns:
        - An `XEBCharacterizationResult` object that contains the calibration result.
        - A `pd.DataFrame` comparing the before and after fidelities.
    """

    pool: Optional['multiprocessing.pool.Pool'] = None
    local_pool = False
    if isinstance(num_workers_or_pool, multiprocessing.pool.Pool):
        pool = num_workers_or_pool  # pragma: no cover
    elif num_workers_or_pool != 0:
        pool = multiprocessing.Pool(num_workers_or_pool if num_workers_or_pool > 0 else None)
        local_pool = True

    fids_df_0, circuits, sampled_df = parallel_xeb_workflow(
        sampler=sampler,
        qubits=qubits,
        entangling_gate=two_qubit_gate,
        n_repetitions=n_repetitions,
        cycle_depths=cycle_depths,
        n_circuits=n_circuits,
        n_combinations=n_combinations,
        random_state=random_state,
        pool=pool,
        tags=tags,
        pairs=pairs,
    )

    if options is None:
        options = xeb_fitting.XEBPhasedFSimCharacterizationOptions(
            characterize_chi=True,
            characterize_gamma=True,
            characterize_zeta=True,
            characterize_theta=False,
            characterize_phi=False,
        ).with_defaults_from_gate(two_qubit_gate)

    p_circuits = [
        xeb_fitting.parameterize_circuit(circuit, options, ops.GateFamily(two_qubit_gate))
        for circuit in circuits
    ]

    result = xeb_fitting.characterize_phased_fsim_parameters_with_xeb_by_pair(
        sampled_df=sampled_df,
        parameterized_circuits=p_circuits,
        cycle_depths=cycle_depths,
        options=options,
        fatol=atol,
        xatol=atol,
        pool=pool,
    )

    before_after = xeb_fitting.before_and_after_characterization(
        fids_df_0, characterization_result=result
    )

    if local_pool:
        assert isinstance(pool, multiprocessing.pool.Pool)
        pool.close()
    return result, before_after


def calibrate_z_phases(
    sampler: 'cirq.Sampler',
    qubits: Optional[Sequence['cirq.GridQubit']] = None,
    two_qubit_gate: 'cirq.Gate' = ops.CZ,
    options: Optional[xeb_fitting.XEBPhasedFSimCharacterizationOptions] = None,
    n_repetitions: int = 10**4,
    n_combinations: int = 10,
    n_circuits: int = 20,
    cycle_depths: Sequence[int] = tuple(np.arange(3, 100, 20)),
    random_state: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None,
    atol: float = 1e-3,
    num_workers_or_pool: Union[int, 'multiprocessing.pool.Pool'] = -1,
    pairs: Optional[Sequence[Tuple['cirq.GridQubit', 'cirq.GridQubit']]] = None,
    tags: Sequence[Any] = (),
) -> Dict[Tuple['cirq.Qid', 'cirq.Qid'], 'cirq.PhasedFSimGate']:
    """Perform z-phase calibration for excitation-preserving gates.

    For a given excitation-preserving two-qubit gate we assume an error model that can be described
    using Z-rotations:
                0: ───Rz(a)───two_qubit_gate───Rz(c)───
                                │
                1: ───Rz(b)───two_qubit_gate───Rz(d)───
    for some angles a, b, c, and d.

    Since the two-qubit gate is a excitation-preserving gate, it can be represented by an FSimGate
    and the effect of rotations turns it into a PhasedFSimGate. Using XEB-data we find the
    PhasedFSimGate parameters that minimize the infidelity of the gate.

    References:
        - https://arxiv.org/abs/2001.08343
        - https://arxiv.org/abs/2010.07965
        - https://arxiv.org/abs/1910.11333

    Args:
        sampler: The quantum engine or simulator to run the circuits.
        qubits: Qubits to use. If none, use all qubits on the sampler's device.
        two_qubit_gate: The entangling gate to use.
        options: The XEB-fitting options. If None, calibrate only the three phase angles
            (chi, gamma, zeta) using the representation of a two-qubit gate as an FSimGate
            for the initial guess.
        n_repetitions: The number of repetitions to use.
        n_combinations: The number of combinations to generate.
        n_circuits: The number of circuits to generate.
        cycle_depths: The cycle depths to use.
        random_state: The random state to use.
        atol: Absolute tolerance to be used by the minimizer.
        num_workers_or_pool: An optional multi-processing pool or number of workers.
            A zero value means no multiprocessing.
            A positive integer value will create a pool with the given number of workers.
            A negative value will create pool with maximum number of workers.
        pairs: Pairs to use. If not specified, use all pairs between adjacent qubits.
        tags: Tags to add to two qubit operations.

    Returns:
        - A dictionary mapping qubit pairs to the calibrated PhasedFSimGates.
    """

    if options is None:
        options = xeb_fitting.XEBPhasedFSimCharacterizationOptions(
            characterize_chi=True,
            characterize_gamma=True,
            characterize_zeta=True,
            characterize_theta=False,
            characterize_phi=False,
        ).with_defaults_from_gate(two_qubit_gate)

    result, _ = z_phase_calibration_workflow(
        sampler=sampler,
        qubits=qubits,
        two_qubit_gate=two_qubit_gate,
        options=options,
        n_repetitions=n_repetitions,
        n_combinations=n_combinations,
        n_circuits=n_circuits,
        cycle_depths=cycle_depths,
        random_state=random_state,
        atol=atol,
        num_workers_or_pool=num_workers_or_pool,
        tags=tags,
        pairs=pairs,
    )

    gates = {}
    for pair, params in result.final_params.items():
        params['theta'] = params.get('theta', options.theta_default or 0)
        params['phi'] = params.get('phi', options.phi_default or 0)
        params['zeta'] = params.get('zeta', options.zeta_default or 0)
        params['chi'] = params.get('chi', options.chi_default or 0)
        params['gamma'] = params.get('gamma', options.gamma_default or 0)
        gates[pair] = ops.PhasedFSimGate(**params)
    return gates


def plot_z_phase_calibration_result(
    before_after_df: 'pd.DataFrame',
    axes: Optional[np.ndarray[Sequence[Sequence['plt.Axes']], np.dtype[np.object_]]] = None,
    pairs: Optional[Sequence[Tuple['cirq.Qid', 'cirq.Qid']]] = None,
    *,
    with_error_bars: bool = False,
) -> np.ndarray[Sequence[Sequence['plt.Axes']], np.dtype[np.object_]]:
    """A helper method to plot the result of running z-phase calibration.

    Note that the plotted fidelity is a statistical estimate of the true fidelity and as a result
    may be outside the [0, 1] range.

    Args:
        before_after_df: The second return object of running `z_phase_calibration_workflow`.
        axes: And ndarray of the axes to plot on.
            The number of axes is expected to be >= number of qubit pairs.
        pairs: If provided, only the given pairs are plotted.
        with_error_bars: Whether to add error bars or not.
            The width of the bar is an upper bound on standard variation of the estimated fidelity.
    """
    if pairs is None:
        pairs = before_after_df.index
    if axes is None:
        # Create a 16x9 rectangle.
        ncols = int(np.ceil(np.sqrt(9 / 16 * len(pairs))))
        nrows = (len(pairs) + ncols - 1) // ncols
        _, axes = plt.subplots(nrows=nrows, ncols=ncols)
    axes = axes if isinstance(axes, np.ndarray) else np.array(axes)
    for pair, ax in zip(pairs, axes.flatten()):
        row = before_after_df.loc[[pair]].iloc[0]
        ax.errorbar(
            row.cycle_depths_0,
            row.fidelities_0,
            yerr=row.layer_fid_std_0 * with_error_bars,
            label='original',
        )
        ax.errorbar(
            row.cycle_depths_0,
            row.fidelities_c,
            yerr=row.layer_fid_std_c * with_error_bars,
            label='calibrated',
        )
        ax.axhline(1, linestyle='--')
        ax.set_xlabel('cycle depth')
        ax.set_ylabel('fidelity estimate')
        ax.set_title('-'.join(str(q) for q in pair))
        ax.legend()
    return axes


def _z_angles(old: ops.PhasedFSimGate, new: ops.PhasedFSimGate) -> Tuple[float, float, float]:
    """Computes a set of possible 3 z-phases that result in the change in gamma, zeta, and chi."""
    # This procedure is the inverse of PhasedFSimGate.from_fsim_rz
    delta_gamma = new.gamma - old.gamma
    delta_zeta = new.zeta - old.zeta
    delta_chi = new.chi - old.chi
    return (-delta_gamma + delta_chi, -delta_gamma - delta_zeta, delta_zeta - delta_chi)


@transformer_api.transformer
class CalibrationTransformer:

    def __init__(
        self,
        target: 'cirq.Gate',
        calibration_map: Dict[Tuple['cirq.Qid', 'cirq.Qid'], 'cirq.PhasedFSimGate'],
    ):
        """Create a CalibrationTransformer.

        The transformer adds 3 ZPowGates around each calibrated gate to cancel the
        effect of z-phases.

        Args:
            target: The target gate. Any gate matching this
                will be replaced based on the content of `calibration_map`.
            calibration_map:
                A map mapping qubit pairs to calibrated gates. This is the output of
                calling `calibrate_z_phases`.
        """
        self.target = target
        if isinstance(target, ops.PhasedFSimGate):
            self.target_as_fsim = target
        elif (gate := ops.PhasedFSimGate.from_matrix(protocols.unitary(target))) is not None:
            self.target_as_fsim = gate
        else:
            raise ValueError(f"{target} is not equivalent to a PhasedFSimGate")
        self.calibration_map = calibration_map

    def __call__(
        self,
        circuit: 'cirq.AbstractCircuit',
        *,
        context: Optional[transformer_api.TransformerContext] = None,
    ) -> 'cirq.Circuit':
        """Adds 3 ZPowGates around each calibrated gate to cancel the effect of Z phases.

        Args:
            circuit: Circuit to transform.
            context: Optional transformer context (not used).

        Returns:
            New circuit with the extra ZPowGates.
        """
        new_moments: List[Union[List[cirq.Operation], 'cirq.Moment']] = []
        for moment in circuit:
            before = []
            after = []
            for op in moment:
                if op.gate != self.target:
                    # not a target.
                    continue
                assert len(op.qubits) == 2
                gate = self.calibration_map.get(op.qubits, None) or self.calibration_map.get(
                    op.qubits[::-1], None
                )
                if gate is None:
                    # no calibration available.
                    continue
                angles = np.array(_z_angles(self.target_as_fsim, gate)) / np.pi
                angles = -angles  # Take the negative to cancel the effect.
                before.append(ops.Z(op.qubits[0]) ** angles[0])
                before.append(ops.Z(op.qubits[1]) ** angles[1])
                after.append(ops.Z(op.qubits[0]) ** angles[2])
            if before:
                new_moments.append(before)
            new_moments.append(moment)
            if after:
                new_moments.append(after)
        return circuits.Circuit.from_moments(*new_moments)
