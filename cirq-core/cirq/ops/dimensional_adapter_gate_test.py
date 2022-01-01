# Copyright 2022 The Cirq Developers
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

import re

import numpy as np
import pytest
import sympy

import cirq


def test_init():
    gate = cirq.DimensionAdapterGate(cirq.X, [(3, (0, 1))])
    assert gate._gate == cirq.X
    assert cirq.qid_shape(gate) == (3,)
    assert gate.subspaces == [(0, 1)]

    with pytest.raises(ValueError, match='Gate qubit count and subspace count must match.'):
        _ = cirq.DimensionAdapterGate(cirq.CX, [(3, (0, 1))])

    with pytest.raises(
        ValueError, match=re.escape('Dimension 3 not large enough for subspace (0, 3) on qubit 0.')
    ):
        _ = cirq.DimensionAdapterGate(cirq.X, [(3, (0, 3))])

    with pytest.raises(
        ValueError, match=re.escape('Subspace (0, 1, 3) does not have consistent step size.')
    ):
        _ = cirq.DimensionAdapterGate(cirq.X, [(5, (0, 1, 3))])

    with pytest.raises(ValueError, match=re.escape('Subspace (0,) is less than 2 dimensions.')):
        _ = cirq.DimensionAdapterGate(cirq.X, [(3, (0,))])

    with pytest.raises(
        ValueError, match=re.escape('Subspace (0, 2) does not match gate dimension 3 on qubit 0.')
    ):
        _ = cirq.DimensionAdapterGate(cirq.IdentityGate(qid_shape=(3,)), [(3, (0, 2))])


@pytest.mark.parametrize('split', [True, False])
def test_simulate_qudits_slices(split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    simulator = cirq.Simulator(split_untangled_states=split)

    circuit = cirq.Circuit(
        cirq.DimensionAdapterGate(cirq.X, [(3, (0, 1))])(q0),
        cirq.DimensionAdapterGate(cirq.X, [(4, (0, 3))])(q1),
    )
    result = simulator.simulate(circuit, qubit_order=[q0, q1])
    expected = np.zeros(12)
    expected[4 * 1 + 3] = 1

    np.testing.assert_almost_equal(result.final_state_vector, expected)
    assert len(result.measurements) == 0

    cirq.testing.assert_has_diagram(
        circuit,
        """
0 (d=3): ───X(subspace [0, 1])───

1 (d=4): ───X(subspace [0, 3])───
""",
        use_unicode_characters=True,
    )


@pytest.mark.parametrize('split', [True, False])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dimensions', [3, 4])
@pytest.mark.parametrize('qubit_count', [3, 4])
def test_simulation_result_is_unitary(split: bool, seed: int, dimensions: int, qubit_count: int):
    prng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(qubit_count)
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=10, op_density=1, random_state=prng
    )
    qubit_map = {q: i for i, q in enumerate(qubits)}
    qudits = cirq.LineQid.range(qubit_count, dimension=dimensions)

    def adapt(op: cirq.Operation) -> cirq.Operation:
        subspaces = [
            (dimensions, tuple(prng.choice(range(dimensions), 2, replace=False))) for _ in op.qubits
        ]
        op_qubits = [qudits[qubit_map[q]] for q in op.qubits]
        return cirq.DimensionAdapterGate(op.gate, subspaces).on(*op_qubits)

    circuit = circuit.map_operations(adapt)
    simulator = cirq.Simulator(split_untangled_states=split)
    result = simulator.simulate(circuit, qubit_order=qudits)
    cirq.validate_normalized_state_vector(
        result.final_state_vector, qid_shape=(dimensions,) * qubit_count
    )


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable(resolve_fn):
    a = sympy.Symbol('a')
    cy = cirq.DimensionAdapterGate(cirq.Y, [(3, slice(0, 2, 1))])
    cya = cirq.DimensionAdapterGate(cirq.YPowGate(exponent=a), [(3, slice(0, 2, 1))])
    assert cirq.is_parameterized(cya)
    assert not cirq.is_parameterized(cy)
    assert resolve_fn(cya, cirq.ParamResolver({'a': 1})) == cy


@pytest.mark.parametrize(
    'gate',
    [
        cirq.X,
        cirq.X ** 0.5,
        cirq.rx(np.pi),
        cirq.rx(np.pi / 2),
        cirq.Z,
        cirq.H,
    ],
)
def test_adapted_gate_is_consistent(gate: cirq.Gate):
    dgate = cirq.DimensionAdapterGate(gate, [(3, (0, 2))])
    cirq.testing.assert_implements_consistent_protocols(dgate)


@pytest.mark.parametrize(
    'gate',
    [
        cirq.CX,
        cirq.CX ** 0.5,
        cirq.CZ,
        cirq.SWAP,
    ],
)
def test_adapted_gate_is_consistent_two_qubits(gate: cirq.Gate):
    dgate = cirq.DimensionAdapterGate(gate, [(3, (0, 2)), (4, (0, 3))])
    cirq.testing.assert_implements_consistent_protocols(dgate)


@pytest.mark.parametrize(
    'gate',
    [
        cirq.X,
        cirq.X ** 0.5,
        cirq.rx(np.pi),
        cirq.rx(np.pi / 2),
        cirq.Z,
        cirq.H,
    ],
)
def test_repr(gate: cirq.Gate):
    dgate = cirq.DimensionAdapterGate(gate, [(3, (0, 2))])
    cirq.testing.assert_equivalent_repr(dgate)


def test_str():
    assert str(cirq.DimensionAdapterGate(cirq.X, [(3, (0, 2))])) == 'X(subspaces=[(0, 2)])'
    assert str(cirq.DimensionAdapterGate(cirq.Z, [(3, (0, 2))])) == 'Z(subspaces=[(0, 2)])'
    assert (
        str(cirq.DimensionAdapterGate(cirq.X ** 0.5, [(3, (0, 2))])) == 'X**0.5(subspaces=[(0, 2)])'
    )
    assert (
        str(cirq.DimensionAdapterGate(cirq.X, [(3, (0, 2))]) ** 0.5) == 'X**0.5(subspaces=[(0, 2)])'
    )
