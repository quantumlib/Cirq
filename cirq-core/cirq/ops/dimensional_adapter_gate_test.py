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
from typing import Mapping

import numpy as np
import pytest
import sympy

import cirq


def test_init():
    gate = cirq.DimensionAdapterGate(cirq.X, [(3, (0, 1))])
    assert gate._gate == cirq.X
    assert cirq.qid_shape(gate) == (3,)
    assert gate.subspaces == [(0, 1)]

    gate = cirq.DimensionAdapterGate(cirq.X, [(4, slice(2, None, -2))])
    assert gate._gate == cirq.X
    assert cirq.qid_shape(gate) == (4,)
    assert gate.subspaces == [(2, 0)]

    gate = cirq.DimensionAdapterGate(cirq.CX, [(3, (0, 1)), (4, (2, 0))])
    assert gate._gate == cirq.CX
    assert cirq.qid_shape(gate) == (3, 4)
    assert gate.subspaces == [(0, 1), (2, 0)]

    gate = cirq.DimensionAdapterGate(cirq.CX, [(3, (0, 1)), (4, slice(2, None, -2))])
    assert gate._gate == cirq.CX
    assert cirq.qid_shape(gate) == (3, 4)
    assert gate.subspaces == [(0, 1), (2, 0)]

    id_gate = cirq.IdentityGate(2, (3, 4))
    gate = cirq.DimensionAdapterGate(id_gate, [(5, (4, 2, 0)), (5, (1, 2, 3, 4))])
    assert gate._gate == id_gate
    assert cirq.qid_shape(gate) == (5, 5)
    assert gate.subspaces == [(4, 2, 0), (1, 2, 3, 4)]

    gate = cirq.DimensionAdapterGate(id_gate, [(5, slice(4, None, -2)), (6, slice(1, 5))])
    assert gate._gate == id_gate
    assert cirq.qid_shape(gate) == (5, 6)
    assert gate.subspaces == [(4, 2, 0), (1, 2, 3, 4)]

    with pytest.raises(ValueError, match='Gate qubit count and subspace count must match.'):
        _ = cirq.DimensionAdapterGate(cirq.CX, [(3, (0, 1))])

    with pytest.raises(ValueError, match=re.escape('slice step cannot be zero')):
        _ = cirq.DimensionAdapterGate(cirq.X, [(3, (1, 1))])

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
@pytest.mark.parametrize('q0_dim', [3, 4])
@pytest.mark.parametrize('q1_dim', [3, 4])
def test_simulate(split: bool, q0_dim: int, q1_dim: int):
    # We put a pair of zero qudits into a target state by applying X's on the (0, target) subspaces
    q0, q1 = cirq.LineQid.for_qid_shape((q0_dim, q1_dim))
    for q0_target in range(1, q0_dim):
        for q1_target in range(1, q1_dim):
            circuit = cirq.Circuit(
                cirq.DimensionAdapterGate(cirq.X, [(q0_dim, (0, q0_target))])(q0),
                cirq.DimensionAdapterGate(cirq.X, [(q1_dim, (0, q1_target))])(q1),
            )
            simulator = cirq.Simulator(split_untangled_states=split)
            result = simulator.simulate(circuit, qubit_order=[q0, q1]).final_state_vector
            expected = np.zeros((q0_dim, q1_dim))
            expected[q0_target, q1_target] = 1
            np.testing.assert_allclose(result.reshape((q0_dim, q1_dim)), expected)


def test_simulate_inverted_condition():
    # CX(q0, q1) on a flipped subspace should be equivalent to X(q0) CX(q0, q1) X(q0)
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.DimensionAdapterGate(cirq.CX, [(2, (1, 0)), (2, (0, 1))])(q0, q1))
    reference_circuit = cirq.Circuit(cirq.X(q0), cirq.CX(q0, q1), cirq.X(q0))
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    reference_result = simulator.simulate(reference_circuit)
    np.testing.assert_allclose(result.final_state_vector, reference_result.final_state_vector)


def test_adapted_condition_depends_only_on_true_subspace():
    # We put q0 into superposition of |0> and |2>
    q0 = cirq.LineQid(0, 3)
    q1 = cirq.LineQubit(1)
    h_02 = cirq.DimensionAdapterGate(cirq.H, [(3, (0, 2))])

    # Then for the CX gate, the control bit can be on (0, 2) or (1, 2) subspace and it should
    # produce the same result. The fact that the 0 subspace and the 1 subspace differ doesn't
    # matter because they're the "zero" of the control.
    zero_circuit = cirq.Circuit(
        h_02(q0), cirq.DimensionAdapterGate(cirq.CX, [(3, (0, 2)), (2, (0, 1))])(q0, q1)
    )
    one_circuit = cirq.Circuit(
        h_02(q0), cirq.DimensionAdapterGate(cirq.CX, [(3, (1, 2)), (2, (0, 1))])(q0, q1)
    )
    simulator = cirq.Simulator()
    zero_result = simulator.simulate(zero_circuit)
    one_result = simulator.simulate(one_circuit)
    np.testing.assert_allclose(zero_result.final_state_vector, one_result.final_state_vector)


def test_diagram():
    q0, q1 = cirq.LineQid.for_qid_shape((3, 4))
    circuit = cirq.Circuit(
        cirq.DimensionAdapterGate(cirq.X, [(3, (0, 1))])(q0),
        cirq.DimensionAdapterGate(cirq.X, [(4, (0, 3))])(q1),
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0 (d=3): ───X(subspace [0, 1])───

1 (d=4): ───X(subspace [0, 3])───
""",
        use_unicode_characters=True,
    )


def test_diagram_complete_subspace_not_annotated():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.DimensionAdapterGate(cirq.CX, [(2, (1, 0)), (2, (0, 1))])(q0, q1))
    cirq.testing.assert_has_diagram(
        circuit,
        """
0: ───@(subspace [1, 0])───
      │
1: ───X────────────────────
""",
        use_unicode_characters=True,
    )


@pytest.mark.parametrize('split', [True, False])
@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dimensions', [3, 4])
@pytest.mark.parametrize('qubit_count', [3, 4])
def test_simulation_result_is_valid(split: bool, seed: int, dimensions: int, qubit_count: int):
    prng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(qubit_count)
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=50, op_density=1, random_state=prng
    )
    qubit_map: Mapping[cirq.Qid, int] = {q: i for i, q in enumerate(qubits)}
    qudits = cirq.LineQid.range(qubit_count, dimension=dimensions)

    def adapt(op: cirq.Operation) -> cirq.Operation:
        gate = op.gate
        assert gate is not None
        subspaces = [
            (dimensions, tuple(prng.choice(range(dimensions), 2, replace=False))) for _ in op.qubits
        ]
        op_qubits = [qudits[qubit_map[q]] for q in op.qubits]
        return cirq.DimensionAdapterGate(gate, subspaces).on(*op_qubits)

    circuit = circuit.map_operations(adapt)
    simulator = cirq.Simulator(split_untangled_states=split)
    result = simulator.simulate(circuit, qubit_order=qudits)
    cirq.validate_normalized_state_vector(
        result.final_state_vector, qid_shape=(dimensions,) * qubit_count
    )


@pytest.mark.parametrize('seed', [0, 1, 2])
@pytest.mark.parametrize('dimensions', [3, 4])
@pytest.mark.parametrize('qubit_count', [3, 4])
def test_unitary(seed: int, dimensions: int, qubit_count: int):
    prng = np.random.RandomState(seed)
    qubits = cirq.LineQubit.range(qubit_count)
    circuit = cirq.testing.random_circuit(
        qubits=qubits, n_moments=50, op_density=1, random_state=prng
    )
    qubit_map: Mapping[cirq.Qid, int] = {q: i for i, q in enumerate(qubits)}
    qudits = cirq.LineQid.range(qubit_count, dimension=dimensions)

    def adapt(op: cirq.Operation) -> cirq.Operation:
        gate = op.gate
        assert gate is not None
        subspaces = [
            (dimensions, tuple(prng.choice(range(dimensions), 2, replace=False))) for _ in op.qubits
        ]
        op_qubits = [qudits[qubit_map[q]] for q in op.qubits]
        return cirq.DimensionAdapterGate(gate, subspaces).on(*op_qubits)

    circuit = circuit.map_operations(adapt)
    result = cirq.unitary(circuit)
    assert cirq.is_unitary(result)


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
