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

import cirq

from cirq.experiments.z_phase_calibration import calibrate_z_phases
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions

_ANGLES = ['theta', 'phi', 'chi', 'zeta', 'gamma']


def _create_tests(n, seed):
    rng = np.random.default_rng(seed)
    angles = (rng.random((n, 5)) * 2 - 1) * np.pi
    # Add errors to the first 2 angles (theta and phi).
    # The errors for theta and phi are in the union (-2, -1) U (1, 2).
    # This is because we run the tests with few repetitions so a small error might not get fixed.
    error = np.concatenate(
        [(rng.random((n, 2)) + 1) * rng.choice([-1, 1], (n, 2)), np.zeros((n, 3))], axis=-1
    )
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


@pytest.mark.parametrize(['angles', 'error'], _create_tests(n=10, seed=32432432))
def test_calibrate_z_phases(angles, error):

    original_gate = cirq.PhasedFSimGate(**{k: v for k, v in zip(_ANGLES, angles)})
    actual_gate = cirq.PhasedFSimGate(**{k: v + e for k, v, e in zip(_ANGLES, angles, error)})

    options = XEBPhasedFSimCharacterizationOptions(
        **{f'{n}_default': t for n, t in zip(_ANGLES, angles)},
        characterize_chi=False,
        characterize_gamma=False,
        characterize_phi=True,
        characterize_theta=True,
        characterize_zeta=False,
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
