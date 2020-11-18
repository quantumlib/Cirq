# Copyright 2018 The Cirq Developers
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

from typing import Optional

import pytest
import numpy as np

import cirq


def test_circuit_gate():
    a, b, c = cirq.LineQubit.range(3)

    g = cirq.CircuitGate(cirq.CZ(a, b))
    assert cirq.num_qubits(g) == 2

    _ = g.on(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g.on(a, c, b)

    _ = g(a, c)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g(a)
    with pytest.raises(ValueError, match='Wrong number'):
        _ = g(c, b, a)
    with pytest.raises(ValueError, match='Wrong shape'):
        _ = g(a, b.with_dimension(3))

    assert g.controlled(0) is g

    gn = cirq.CircuitGate(cirq.CZ(a, b), name='named_gate')
    assert g.circuit == gn.circuit
    assert g != gn

    gn = cirq.CircuitGate(cirq.CZ(a, b), exp_modulus=2)
    assert g.circuit == gn.circuit
    assert g != gn


def test_circuit_op():
    a, b, c = cirq.LineQubit.range(3)
    g = cirq.CircuitGate(cirq.X(a))
    op = g(a)
    assert op.controlled_by() is op
    controlled_op = op.controlled_by(b, c)
    assert controlled_op.sub_operation == op
    assert controlled_op.controls == (b, c)


def test_circuit_op_validate():
    cg = cirq.CircuitGate(cirq.X(cirq.NamedQubit('placeholder')))
    op = cg.on(cirq.LineQid(0, 2))
    cg2 = cirq.CircuitGate(cirq.CNOT(*cirq.LineQubit.range(2)))
    op2 = cg2.on(*cirq.LineQid.range(2, dimension=2))
    op.validate_args([cirq.LineQid(1, 2)])  # Valid
    op2.validate_args(cirq.LineQid.range(1, 3, dimension=2))  # Valid
    with pytest.raises(ValueError, match='Wrong shape'):
        op.validate_args([cirq.LineQid(1, 9)])
    with pytest.raises(ValueError, match='Wrong number'):
        op.validate_args([cirq.LineQid(1, 2), cirq.LineQid(2, 2)])
    with pytest.raises(ValueError, match='Duplicate'):
        op2.validate_args([cirq.LineQid(1, 2), cirq.LineQid(1, 2)])


def test_default_validation_and_inverse():
    a, b = cirq.LineQubit.range(2)
    cg = cirq.CircuitGate(cirq.Z(a), cirq.S(b), cirq.X(a))

    i = cg**-1
    assert i**-1 == cg
    assert cg**-1 == i
    assert cirq.Circuit(cirq.decompose(i)) == cirq.Circuit(
        cirq.X(a),
        cirq.S(b)**-1, cirq.Z(a))
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(i),
                                                    cirq.unitary(cg).conj().T,
                                                    atol=1e-8)

    cirq.testing.assert_implements_consistent_protocols(
        i, local_vals={'CircuitGate': cirq.CircuitGate})


def test_default_inverse():
    qubits = cirq.LineQubit.range(3)
    cg = cirq.CircuitGate(cirq.X(q) for q in qubits)

    assert cirq.inverse(cg, None) is not None
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(cg))
    cirq.testing.assert_has_consistent_qid_shape(
        cirq.inverse(cg.on(*cirq.LineQubit.range(3))))


def test_no_inverse_if_not_unitary():
    cg = cirq.CircuitGate(cirq.amplitude_damp(0.5).on(cirq.LineQubit(0)))
    assert cirq.inverse(cg, None) is None


def test_default_qudit_inverse():
    q = cirq.LineQid.for_qid_shape((1, 2, 3))
    cg = cirq.CircuitGate(
        cirq.IdentityGate(qid_shape=(1,)).on(q[0]),
        (cirq.X**0.1).on(q[1]),
        cirq.IdentityGate(qid_shape=(3,)).on(q[2]),
    )

    assert cirq.qid_shape(cg) == (1, 2, 3)
    assert cirq.qid_shape(cirq.inverse(cg, None)) == (1, 2, 3)
    cirq.testing.assert_has_consistent_qid_shape(cirq.inverse(cg))


def test_circuit_gate_shape():
    shape_gate = cirq.CircuitGate(
        cirq.IdentityGate(qid_shape=(q.dimension,)).on(q)
        for q in cirq.LineQid.for_qid_shape((1, 2, 3, 4)))
    assert cirq.qid_shape(shape_gate) == (1, 2, 3, 4)
    assert cirq.num_qubits(shape_gate) == 4
    assert shape_gate.num_qubits() == 4

    qubit_gate = cirq.CircuitGate(cirq.I(q) for q in cirq.LineQubit.range(3))
    assert cirq.qid_shape(qubit_gate) == (2, 2, 2)
    assert cirq.num_qubits(qubit_gate) == 3
    assert qubit_gate.num_qubits() == 3


def test_circuit_gate_json_dict():
    cg = cirq.CircuitGate(cirq.X(cirq.LineQubit(0)))
    assert cg._json_dict_() == {
        'cirq_type': 'CircuitGate',
        'circuit': cg.circuit,
        'name': cg.name,
        'exp_modulus': cg.exp_modulus,
    }


def test_string_format():
    x, y, z = cirq.LineQubit.range(3)

    cg = cirq.CircuitGate(cirq.X(x), cirq.H(y), cirq.CX(y, z),
                          cirq.measure(x, y, z, key='m'))

    assert str(cg) == """\
CircuitGate:
[ 0: ───X───────M('m')─── ]
[               │         ]
[ 1: ───H───@───M──────── ]
[           │   │         ]
[ 2: ───────X───M──────── ]"""


# Test CircuitGates in Circuits.


def test_circuit_gate_and_gate():
    # A circuit gate and gate in parallel.
    q = cirq.LineQubit.range(3)
    ops_circuit = cirq.Circuit(
        cirq.CircuitGate(
            cirq.X(q[0]),
            cirq.H(q[2]),
            cirq.CZ(q[0], q[2]),
        ).on(q[0], q[2]),
        cirq.X(q[1]),
    )
    flat_circuit = cirq.Circuit(
        cirq.X(q[0]),
        cirq.H(q[2]),
        cirq.CZ(q[0], q[2]),
        cirq.X(q[1]),
    )

    assert np.allclose(ops_circuit.unitary(), flat_circuit.unitary())


def test_parallel_circuit_gates():
    # Two circuit gates in parallel.
    q = cirq.LineQubit.range(3)
    ops_circuit = cirq.Circuit(
        cirq.CircuitGate(
            cirq.X(q[0]),
            cirq.H(q[2]),
            cirq.CZ(q[0], q[2]),
        ).on(q[0], q[2]),
        cirq.CircuitGate(
            cirq.X(q[1]),
            cirq.H(q[1]),
        ).on(q[1]),
    )
    flat_circuit = cirq.Circuit(
        cirq.X(q[0]),
        cirq.H(q[2]),
        cirq.CZ(q[0], q[2]),
        cirq.X(q[1]),
        cirq.H(q[1]),
    )

    assert np.allclose(ops_circuit.unitary(), flat_circuit.unitary())


def test_nested_circuit_gates():
    # A circuit gate inside another circuit gate.
    q = cirq.LineQubit.range(3)
    ops_circuit = cirq.Circuit(
        cirq.CircuitGate(
            cirq.CircuitGate(cirq.X(q[0]), cirq.H(q[0])).on(q[0]),
            cirq.H(q[2]),
            cirq.CZ(q[0], q[2]),
        ).on(q[0], q[2]),
        cirq.CircuitGate(cirq.X(q[1]), cirq.H(q[1])).on(q[1]),
    )
    flat_circuit = cirq.Circuit(
        cirq.X(q[0]),
        cirq.H(q[0]),
        cirq.H(q[2]),
        cirq.CZ(q[0], q[2]),
        cirq.X(q[1]),
        cirq.H(q[1]),
    )

    assert np.allclose(ops_circuit.unitary(), flat_circuit.unitary())


def test_circuit_gate_with_qubits():
    # Reassigning qubits of a circuit gate using on().
    q = cirq.LineQubit.range(3)
    ops_circuit = cirq.Circuit(
        cirq.CircuitGate(
            cirq.X(q[0]),
            cirq.H(q[1]),
            cirq.CZ(q[0], q[1]),
        ).on(q[0], q[2]),
        cirq.X(q[1]),
    )
    flat_circuit = cirq.Circuit(
        cirq.X(q[0]),
        cirq.H(q[2]),
        cirq.CZ(q[0], q[2]),
        cirq.X(q[1]),
    )

    assert np.allclose(ops_circuit.unitary(), flat_circuit.unitary())


def test_circuit_gate_gate_collision():
    # A circuit gate and a gate on the same qubit(s) at the same time
    # should produce an error.
    q = cirq.LineQubit.range(3)
    with pytest.raises(ValueError):
        cirq.Circuit(
            cirq.Moment(
                cirq.CircuitGate(
                    cirq.X(q[0]),
                    cirq.H(q[1]),
                    cirq.CZ(q[0], q[1]),
                ).on(q[0], q[1]),
                cirq.X(q[1]),
            ),)


def test_multi_circuit_gate_collision():
    # Two circuit gates on the same qubit(s) at the same time should
    # produce an error.
    q = cirq.LineQubit.range(3)
    with pytest.raises(ValueError):
        cirq.Circuit(
            cirq.Moment(
                cirq.CircuitGate(
                    cirq.X(q[0]),
                    cirq.H(q[1]),
                    cirq.CZ(q[0], q[1]),
                ).on(q[0], q[1]),
                cirq.CircuitGate(
                    cirq.X(q[2]),
                    cirq.H(q[1]),
                    cirq.CZ(q[2], q[1]),
                ).on(q[1], q[2]),
            ),)


def test_no_exp_modulus_for_measurement():
    with pytest.raises(ValueError):
        _ = cirq.CircuitGate(cirq.measure(cirq.LineQubit(0)), exp_modulus=2)


def test_terminal_matches():
    a, b = cirq.LineQubit.range(2)
    cg = cirq.CircuitGate(
        cirq.H(a),
        cirq.measure(b, key='m1'),
    )

    c = cirq.Circuit(cirq.X(a), cg.on(a, b))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.X(b), cg.on(a, b))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.measure(a), cg.on(a, b))
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(cirq.measure(b), cg.on(a, b))
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(cg.on(a, b), cirq.X(a))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cg.on(a, b), cirq.X(b))
    assert not c.are_all_measurements_terminal()

    c = cirq.Circuit(cg.on(a, b), cirq.measure(a))
    assert c.are_all_measurements_terminal()

    c = cirq.Circuit(cg.on(a, b), cirq.measure(b))
    assert not c.are_all_measurements_terminal()


def test_nonterminal_in_circuit_gate():
    a, b = cirq.LineQubit.range(2)
    cg = cirq.CircuitGate(
        cirq.H(a),
        cirq.measure(b, key='m1'),
        cirq.X(b),
    )

    c = cirq.Circuit(cirq.X(a), cg.on(a, b))
    assert not c.are_all_measurements_terminal()


# Demonstrate applications.


def test_simulate_circuit_gate():
    q = cirq.LineQubit.range(4)
    ops_circuit = cirq.Circuit(
        cirq.Moment(
            cirq.X(q[3]),
            cirq.CircuitGate(
                cirq.H(q[0]),
                cirq.CX(q[0], q[2]),
            ).on(q[0], q[2]),
            cirq.X(q[1]),
        ),)
    flat_circuit = cirq.Circuit(
        cirq.H(q[0]),
        cirq.CX(q[0], q[2]),
        cirq.X(q[1]),
        cirq.X(q[3]),
    )

    simulator = cirq.Simulator()
    ops_result = simulator.simulate(ops_circuit)
    flat_result = simulator.simulate(flat_circuit)
    assert cirq.equal_up_to_global_phase(ops_result.state_vector(),
                                         flat_result.state_vector())


def test_point_optimizer():

    class Opty(cirq.PointOptimizer):

        def optimization_at(self, circuit: 'cirq.Circuit', index: int,
                            op: 'cirq.Operation'
                           ) -> Optional[cirq.PointOptimizationSummary]:
            if isinstance(op.gate, cirq.CircuitGate):
                base_circuit = op.gate.circuit.unfreeze()
                Opty().optimize_circuit(base_circuit)
                return cirq.PointOptimizationSummary(
                    clear_span=1,
                    clear_qubits=op.qubits,
                    new_operations=cirq.CircuitGate(base_circuit).on(
                        *op.qubits))
            if isinstance(op.gate, cirq.CZPowGate):
                return cirq.PointOptimizationSummary(
                    clear_span=1,
                    clear_qubits=op.qubits,
                    new_operations=cirq.CircuitGate(
                        cirq.CZ(*op.qubits), cirq.X.on_each(*op.qubits),
                        cirq.X.on_each(*op.qubits),
                        cirq.CZ(*op.qubits)).on(*op.qubits))
            return None

    cg_qubits = [cirq.GridQubit(5, 2), cirq.GridQubit(5, 3)]
    circuit = cirq.Circuit(
        cirq.CZ(*cg_qubits),
        cirq.X(cirq.GridQubit(6, 2)),
    )

    Opty().optimize_circuit(circuit)
    assert circuit == cirq.Circuit(
        cirq.CircuitGate(
            cirq.CZ(*cg_qubits),
            cirq.X.on_each(*cg_qubits),
            cirq.X.on_each(*cg_qubits),
            cirq.CZ(*cg_qubits),
        ).on(*cg_qubits),
        cirq.X(cirq.GridQubit(6, 2)),
    )

    Opty().optimize_circuit(circuit)
    assert circuit == cirq.Circuit(
        cirq.CircuitGate(
            cirq.CircuitGate(cirq.CZ(*cg_qubits), cirq.X.on_each(*cg_qubits),
                             cirq.X.on_each(*cg_qubits),
                             cirq.CZ(*cg_qubits)).on(*cg_qubits),
            cirq.X.on_each(*cg_qubits),
            cirq.X.on_each(*cg_qubits),
            cirq.CircuitGate(cirq.CZ(*cg_qubits), cirq.X.on_each(*cg_qubits),
                             cirq.X.on_each(*cg_qubits),
                             cirq.CZ(*cg_qubits)).on(*cg_qubits),
        ).on(*cg_qubits),
        cirq.X(cirq.GridQubit(6, 2)),
    )
