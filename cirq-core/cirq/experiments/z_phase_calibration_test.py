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

import pytest
import numpy as np
import pandas as pd

import cirq

from cirq.experiments.z_phase_calibration import (
    calibrate_z_phases,
    z_phase_calibration_workflow,
    plot_z_phase_calibration_result,
)
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions

_ANGLES = ['theta', 'phi', 'chi', 'zeta', 'gamma']


def _create_tests(n, seed, with_options: bool = False):
    rng = np.random.default_rng(seed)
    angles = (rng.random((n, 5)) * 2 - 1) * np.pi
    # Add errors to the last 3 angles (chi, zeta, gamma).
    # The errors are in the union (-2, -1) U (1, 2).
    # This is because we run the tests with few repetitions so a small error might not get fixed.
    error = np.concatenate(
        [np.zeros((n, 2)), (rng.random((n, 3)) + 1) * rng.choice([-1, 1], (n, 3))], axis=-1
    )
    if with_options:
        options = []
        for _ in range(n):
            v = [False, False, False]
            # Calibrate only one to keep the run time down.
            v[rng.integers(0, 3)] = True
            options.append(
                {
                    'characterize_chi': v[0],
                    'characterize_gamma': v[1],
                    'characterize_zeta': v[2],
                    'characterize_phi': False,
                    'characterize_theta': False,
                }
            )

        return zip(angles, error, options)
    return zip(angles, error)


def _trace_distance(A, B):
    return 0.5 * np.abs(np.linalg.eigvals(A - B)).sum()


class _TestSimulator(cirq.Simulator):
    """A simulator that replaces a specific gate by another."""

    def __init__(self, gate: cirq.Gate, replacement: cirq.Gate, **kwargs):
        super().__init__(**kwargs)
        self.gate = gate
        self.replacement = replacement

    def _core_iterator(
        self,
        circuit: 'cirq.AbstractCircuit',
        sim_state,
        all_measurements_are_terminal: bool = False,
    ):
        new_circuit = cirq.Circuit(
            [
                [op if op.gate != self.gate else self.replacement(*op.qubits) for op in m]
                for m in circuit
            ]
        )
        yield from super()._core_iterator(new_circuit, sim_state, all_measurements_are_terminal)


@pytest.mark.parametrize(
    ['angles', 'error', 'characterization_flags'],
    _create_tests(n=10, seed=32432432, with_options=True),
)
def test_calibrate_z_phases(angles, error, characterization_flags):

    original_gate = cirq.PhasedFSimGate(**{k: v for k, v in zip(_ANGLES, angles)})
    actual_gate = cirq.PhasedFSimGate(**{k: v + e for k, v, e in zip(_ANGLES, angles, error)})

    options = XEBPhasedFSimCharacterizationOptions(
        **{f'{n}_default': t for n, t in zip(_ANGLES, angles)}, **characterization_flags
    )

    sampler = _TestSimulator(original_gate, actual_gate, seed=0)
    qubits = cirq.q(0, 0), cirq.q(0, 1)
    calibrated_gate = calibrate_z_phases(
        sampler,
        qubits,
        original_gate,
        options,
        n_repetitions=10,
        n_combinations=10,
        n_circuits=10,
        cycle_depths=range(3, 10),
    )[qubits]

    initial_unitary = cirq.unitary(original_gate)
    final_unitary = cirq.unitary(calibrated_gate)
    target_unitary = cirq.unitary(actual_gate)
    maximally_mixed_state = np.eye(4) / 2
    dm_initial = initial_unitary @ maximally_mixed_state @ initial_unitary.T.conj()
    dm_final = final_unitary @ maximally_mixed_state @ final_unitary.T.conj()
    dm_target = target_unitary @ maximally_mixed_state @ target_unitary.T.conj()

    original_dist = _trace_distance(dm_initial, dm_target)
    new_dist = _trace_distance(dm_final, dm_target)

    # Either we reduced the error or the error is small enough.
    assert new_dist < original_dist or new_dist < 1e-6


@pytest.mark.parametrize(['angles', 'error'], _create_tests(n=3, seed=32432432))
def test_calibrate_z_phases_no_options(angles, error):

    original_gate = cirq.PhasedFSimGate(**{k: v for k, v in zip(_ANGLES, angles)})
    actual_gate = cirq.PhasedFSimGate(**{k: v + e for k, v, e in zip(_ANGLES, angles, error)})

    sampler = _TestSimulator(original_gate, actual_gate, seed=0)
    qubits = cirq.q(0, 0), cirq.q(0, 1)
    calibrated_gate = calibrate_z_phases(
        sampler,
        qubits,
        original_gate,
        options=None,
        n_repetitions=10,
        n_combinations=10,
        n_circuits=10,
        cycle_depths=range(3, 10),
    )[qubits]

    initial_unitary = cirq.unitary(original_gate)
    final_unitary = cirq.unitary(calibrated_gate)
    target_unitary = cirq.unitary(actual_gate)
    maximally_mixed_state = np.eye(4) / 2
    dm_initial = initial_unitary @ maximally_mixed_state @ initial_unitary.T.conj()
    dm_final = final_unitary @ maximally_mixed_state @ final_unitary.T.conj()
    dm_target = target_unitary @ maximally_mixed_state @ target_unitary.T.conj()

    original_dist = _trace_distance(dm_initial, dm_target)
    new_dist = _trace_distance(dm_final, dm_target)

    # Either we reduced the error or the error is small enough.
    assert new_dist < original_dist or new_dist < 1e-6


@pytest.mark.parametrize(['angles', 'error'], _create_tests(n=3, seed=32432432))
def test_calibrate_z_phases_workflow_no_options(angles, error):

    original_gate = cirq.PhasedFSimGate(**{k: v for k, v in zip(_ANGLES, angles)})
    actual_gate = cirq.PhasedFSimGate(**{k: v + e for k, v, e in zip(_ANGLES, angles, error)})

    sampler = _TestSimulator(original_gate, actual_gate, seed=0)
    qubits = cirq.q(0, 0), cirq.q(0, 1)
    result, _ = z_phase_calibration_workflow(
        sampler,
        qubits,
        original_gate,
        options=None,
        n_repetitions=1,
        n_combinations=1,
        n_circuits=1,
        cycle_depths=(1, 2),
    )

    for params in result.final_params.values():
        assert 'zeta' in params
        assert 'chi' in params
        assert 'gamma' in params
        assert 'phi' not in params
        assert 'theta' not in params


def test_plot_z_phase_calibration_result():
    df = pd.DataFrame()
    qs = cirq.q(0, 0), cirq.q(0, 1), cirq.q(0, 2)
    df.index = [qs[:2], qs[-2:]]
    df['cycle_depths_0'] = [[1, 2, 3]] * 2
    df['fidelities_0'] = [[0.9, 0.8, 0.7], [0.6, 0.4, 0.1]]
    df['layer_fid_std_0'] = [0.1, 0.2]
    df['fidelities_c'] = [[0.9, 0.92, 0.93], [0.7, 0.77, 0.8]]
    df['layer_fid_std_c'] = [0.2, 0.3]

    axes = plot_z_phase_calibration_result(before_after_df=df)

    np.testing.assert_allclose(axes[0].lines[0].get_xdata().astype(float), [1, 2, 3])
    np.testing.assert_allclose(axes[0].lines[0].get_ydata().astype(float), [0.9, 0.8, 0.7])
    np.testing.assert_allclose(axes[0].lines[1].get_ydata().astype(float), [0.9, 0.92, 0.93])

    np.testing.assert_allclose(axes[1].lines[0].get_xdata().astype(float), [1, 2, 3])
    np.testing.assert_allclose(axes[1].lines[0].get_ydata().astype(float), [0.6, 0.4, 0.1])
    np.testing.assert_allclose(axes[1].lines[1].get_ydata().astype(float), [0.7, 0.77, 0.8])
