# Copyright 2026 The Cirq Developers
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

from __future__ import annotations

import numpy as np
import pytest

import cirq


def _expected_state_vector(state, qid_shape, axes, subspaces):
    """Brute force reference: keeps the amplitudes whose post-selected digits are in `subspaces`."""
    tensor = np.asarray(state).reshape(qid_shape)
    kept = np.zeros_like(tensor)
    for index in np.ndindex(*qid_shape):
        if tuple(index[axis] for axis in axes) in subspaces:
            kept[index] = tensor[index]
    return (kept / np.linalg.norm(kept)).reshape(-1)


def _expected_density_matrix(rho, qid_shape, axes, subspaces):
    """Brute force reference: P rho P / tr(P rho P) for the projector P onto the subspace."""
    keep = [tuple(index[axis] for axis in axes) in subspaces for index in np.ndindex(*qid_shape)]
    projector = np.diag(np.array(keep, dtype=float))
    projected = projector @ rho @ projector
    return projected / np.trace(projected)


def _random_state_vector(dim, seed):
    rng = np.random.RandomState(seed)
    state = rng.randn(dim) + 1j * rng.randn(dim)
    return state / np.linalg.norm(state)


def _random_density_matrix(dim, seed):
    rng = np.random.RandomState(seed)
    a = rng.randn(dim, dim) + 1j * rng.randn(dim, dim)
    rho = a @ a.conj().T
    return rho / np.trace(rho)


_CASES = [
    ((0,), [(1,)]),
    ((2, 0), [(0, 1), (1, 0)]),
    ((1,), [(0,), (1,)]),
    ((0, 1, 2), [(0, 0, 0), (1, 1, 1), (0, 1, 0)]),
]


def test_init_validation() -> None:
    with pytest.raises(ValueError, match='At least one subspace'):
        _ = cirq.PostSelectionGate((2,), [])
    with pytest.raises(ValueError, match='does not match'):
        _ = cirq.PostSelectionGate((2, 2), [(0,)])
    with pytest.raises(ValueError, match='out of range'):
        _ = cirq.PostSelectionGate((2, 3), [(0, 3)])


def test_equality() -> None:
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.PostSelectionGate((2, 2), [(0, 0), (1, 1)]),
        cirq.PostSelectionGate([2, 2], [[1, 1], [0, 0], [1, 1]]),
    )
    eq.add_equality_group(cirq.PostSelectionGate((2, 2), [(0, 0)]))
    eq.add_equality_group(cirq.PostSelectionGate((3, 3), [(0, 0), (1, 1)]))


def test_repr_and_json() -> None:
    gate = cirq.PostSelectionGate((2, 3), [(0, 1), (1, 2)])
    cirq.testing.assert_equivalent_repr(gate)
    cirq.testing.assert_json_roundtrip_works(gate)


def test_is_neither_unitary_nor_a_channel_nor_a_measurement() -> None:
    gate = cirq.PostSelectionGate((2,), [(1,)])
    assert cirq.qid_shape(gate) == (2,)
    assert cirq.num_qubits(gate) == 1
    assert not cirq.has_unitary(gate)
    assert not cirq.has_kraus(gate)
    assert not cirq.has_mixture(gate)
    assert not cirq.is_measurement(gate)
    cirq.testing.assert_implements_consistent_protocols(
        gate, ignore_decompose_to_default_gateset=True
    )


def test_diagram() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    gate = cirq.PostSelectionGate((2, 2), [(0, 0), (1, 1)])
    cirq.testing.assert_has_diagram(
        cirq.Circuit(gate.on(q0, q1)),
        """
0: ───PostSelect(00|11)───
      │
1: ───PostSelect(00|11)───
""",
    )


@pytest.mark.parametrize('split', [True, False])
@pytest.mark.parametrize('axes,subspaces', _CASES)
def test_state_vector_simulation_matches_brute_force(axes, subspaces, split) -> None:
    qubits = cirq.LineQubit.range(3)
    initial_state = _random_state_vector(8, seed=7)
    gate = cirq.PostSelectionGate((2,) * len(axes), subspaces)
    circuit = cirq.Circuit(gate.on(*(qubits[axis] for axis in axes)))
    simulator = cirq.Simulator(dtype=np.complex128, split_untangled_states=split)
    result = simulator.simulate(circuit, qubit_order=qubits, initial_state=initial_state)
    expected = _expected_state_vector(initial_state, (2, 2, 2), axes, set(subspaces))
    np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-9)


@pytest.mark.parametrize('split', [True, False])
@pytest.mark.parametrize('axes,subspaces', _CASES)
def test_density_matrix_simulation_matches_brute_force(axes, subspaces, split) -> None:
    qubits = cirq.LineQubit.range(3)
    initial_state = _random_density_matrix(8, seed=3)
    gate = cirq.PostSelectionGate((2,) * len(axes), subspaces)
    circuit = cirq.Circuit(gate.on(*(qubits[axis] for axis in axes)))
    simulator = cirq.DensityMatrixSimulator(dtype=np.complex128, split_untangled_states=split)
    result = simulator.simulate(circuit, qubit_order=qubits, initial_state=initial_state)
    expected = _expected_density_matrix(initial_state, (2, 2, 2), axes, set(subspaces))
    np.testing.assert_allclose(result.final_density_matrix, expected, atol=1e-9)


def test_qutrits() -> None:
    qutrits = cirq.LineQid.range(2, dimension=3)
    circuit = cirq.Circuit(cirq.PostSelectionGate((3, 3), [(0, 0), (1, 2)]).on(*qutrits))
    initial_state = np.ones(9) / 3
    expected = np.zeros(9)
    expected[[0, 5]] = np.sqrt(0.5)  # |00> and |12>
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit, initial_state=initial_state)
    np.testing.assert_allclose(result.final_state_vector, expected, atol=1e-9)
    rho = np.outer(initial_state, initial_state)
    dm_result = cirq.DensityMatrixSimulator(dtype=np.complex128).simulate(
        circuit, initial_state=rho
    )
    np.testing.assert_allclose(
        dm_result.final_density_matrix, np.outer(expected, expected), atol=1e-9
    )


def test_parity_post_selection_prepares_a_bell_state() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    gate = cirq.PostSelectionGate((2, 2), [(0, 0), (1, 1)])
    circuit = cirq.Circuit(cirq.H.on_each(q0, q1), gate.on(q0, q1))
    result = cirq.Simulator(dtype=np.complex128).simulate(circuit)
    np.testing.assert_allclose(
        result.final_state_vector, [np.sqrt(0.5), 0, 0, np.sqrt(0.5)], atol=1e-6
    )


@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('simulator_type', [cirq.Simulator, cirq.DensityMatrixSimulator])
def test_impossible_post_selection_raises(simulator_type, dtype) -> None:
    q = cirq.LineQubit(0)
    gate = cirq.PostSelectionGate((2,), [(1,)])
    # A fresh qubit is exactly |0> (no prior gates, so no accumulated floating-point
    # error), giving exactly zero overlap with |1> regardless of dtype or platform.
    circuit = cirq.Circuit(gate.on(q))
    with pytest.raises(ValueError, match='no support'):
        simulator_type(dtype=dtype).simulate(circuit)


def test_failed_post_selection_leaves_the_state_unchanged() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    for state in (
        cirq.StateVectorSimulationState(qubits=[q0, q1], initial_state=1),
        cirq.DensityMatrixSimulationState(qubits=[q0, q1], initial_state=1),
    ):
        before = state.target_tensor.copy()
        with pytest.raises(ValueError, match='no support'):
            state.post_select([q0], [(1,)])
        np.testing.assert_array_equal(state.target_tensor, before)
        # The buffers must still be usable after a failed and after a successful projection.
        state.post_select([q0], [(0,)])
        np.testing.assert_allclose(state.target_tensor, before)


@pytest.mark.parametrize(
    'simulator',
    [
        cirq.Simulator(),
        cirq.Simulator(split_untangled_states=False),
        cirq.DensityMatrixSimulator(),
        cirq.DensityMatrixSimulator(split_untangled_states=False),
    ],
)
def test_run_only_sees_the_post_selected_outcome(simulator) -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(
        cirq.H(q), cirq.PostSelectionGate((2,), [(1,)]).on(q), cirq.X(q), cirq.measure(q, key='m')
    )
    result = simulator.run(circuit, repetitions=20)
    assert np.all(result.measurements['m'] == 0)


@pytest.mark.parametrize('simulator_type', [cirq.Simulator, cirq.DensityMatrixSimulator])
def test_post_selection_after_measurement_of_an_entangled_qubit(simulator_type) -> None:
    # q0 and q1 start entangled (a Bell pair). q0 is measured, then q1 -- which was never
    # itself touched by that measurement operation -- is post-selected. The two qubits'
    # outcomes must stay correlated as if the circuit were simulated one moment at a time:
    # post-selecting q1 onto |1> can only succeed in the branch where q0 measured 1, and
    # must raise (leaving q0's reported outcome only ever 1, never 0) in the branch where
    # q0 measured 0, where q1 has no support on |1>. A simulator that is free to compute a
    # post-selection once and reuse it before it knows which branch a later, correlated
    # measurement lands in would instead see every repetition succeed with the same q0.
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.H(q0)]),
            cirq.Moment([cirq.CNOT(q0, q1)]),
            cirq.Moment([cirq.measure(q0, key='m0')]),
            cirq.Moment([cirq.PostSelectionGate((2,), [(1,)]).on(q1)]),
        ]
    )
    successes = 0
    failures = 0
    for seed in range(40):
        try:
            result = simulator_type(seed=seed).simulate(circuit)
        except ValueError:
            failures += 1
            continue
        successes += 1
        assert result.measurements['m0'][0] == 1
    # With 40 independent, roughly 50/50 trials, seeing zero of either outcome would be
    # astronomically unlikely if the two qubits are being handled correctly.
    assert successes > 0
    assert failures > 0


def test_unsupported_simulator() -> None:
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.PostSelectionGate((2,), [(0,)]).on(q))
    with pytest.raises(NotImplementedError, match='does not support post-selection'):
        cirq.CliffordSimulator().simulate(circuit)
