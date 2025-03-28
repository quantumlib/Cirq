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

"""Tests for SycamoreTargetGateset."""

import numpy as np
import pytest

import cirq
import cirq_google
from cirq_google.transformers.target_gatesets import sycamore_gateset

# pylint: disable=line-too-long


def test_merge_swap_rzz_and_2q_unitaries():
    q = cirq.LineQubit.range(3)
    c_orig = cirq.Circuit(
        cirq.SWAP(*q[:2]),
        cirq.ZZ(*q[:2]) ** 0.5,
        cirq.ZZ(*q[:2]) ** 0.25,
        cirq.SWAP(*q[:2]),
        cirq.SWAP(q[0], q[2]).with_tags("ignore"),
        cirq.ZZ(q[0], q[2]) ** 0.75,
        cirq.Moment(cirq.H.on_each(*q)),
        cirq.CNOT(q[0], q[2]),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(cirq.CNOT(*q[0:2]), cirq.H(q[0]), cirq.CZ(*q[:2]))
        ),
        cirq.CNOT(*q[1:3]),
        cirq.X(q[0]),
        cirq.ZZ(*q[:2]) ** 0.15,
        cirq.SWAP(*q[:2]),
        cirq.Moment(cirq.X(q[0]).with_tags("ignore"), cirq.Y(q[1])),
        cirq.CNOT(*q[:2]),
        strategy=cirq.InsertStrategy.NEW,
    )
    cirq.testing.assert_has_diagram(
        c_orig,
        '''
                                                               [ 0: ───@───H───@─── ]
0: ───×───ZZ───────ZZ────────×───×[ignore]───ZZ────────H───@───[       │       │    ]───────X───ZZ────────×───X[ignore]───@───
      │   │        │         │   │           │             │   [ 1: ───X───────@─── ]           │         │               │
      │   │        │         │   │           │             │   │                                │         │               │
1: ───×───ZZ^0.5───ZZ^0.25───×───┼───────────┼─────────H───┼───#2───────────────────────@───────ZZ^0.15───×───Y───────────X───
                                 │           │             │                            │
2: ──────────────────────────────×───────────ZZ^0.75───H───X────────────────────────────X─────────────────────────────────────
''',
    )

    c_new = sycamore_gateset.merge_swap_rzz_and_2q_unitaries(
        c_orig,
        context=cirq.TransformerContext(tags_to_ignore=("ignore",)),
        merged_swap_rzz_tag='swap_rzz',
        merged_2q_component_tag='2q_component',
    )
    cirq.testing.assert_has_diagram(
        c_new,
        '''
                                                                                                                                                        [           [ 0: ───@───H───@─── ]        ]
      [ 0: ───×───ZZ─────── ]                 [ 0: ───ZZ────────×─── ]                             [ 0: ───ZZ────────H───@─── ]                         [ 0: ───────[       │       │    ]───X─── ]                         [ 0: ───ZZ────────×─── ]                             [ 0: ───────@─── ]
0: ───[       │   │         ]─────────────────[       │         │    ]─────────────────×[ignore]───[       │             │    ]─────────────────────────[           [ 1: ───X───────@─── ]        ]─────────────────────────[       │         │    ]─────────────────X[ignore]───[           │    ]─────────────────
      [ 1: ───×───ZZ^0.5─── ][swap_rzz]       [ 1: ───ZZ^0.25───×─── ][swap_rzz]       │           [ 2: ───ZZ^0.75───H───X─── ][2q_component]           [           │                             ]                         [ 1: ───ZZ^0.15───×─── ][swap_rzz]                   [ 1: ───Y───X─── ][2q_component]
      │                                       │                                        │           │                                                    [ 1: ───H───#2─────────────────────────── ][2q_component]           │                                                    │
      │                                       │                                        │           │                                                    │                                                                   │                                                    │
1: ───#2──────────────────────────────────────#2───────────────────────────────────────┼───────────┼────────────────────────────────────────────────────#2──────────────────────────────────────────────────────────@───────#2───────────────────────────────────────────────────#2─────────────────────────────────
                                                                                       │           │                                                                                                                │
2: ────────────────────────────────────────────────────────────────────────────────────×───────────#2───────────────────────────────────────────────────────────────────────────────────────────────────────────────X───────────────────────────────────────────────────────────────────────────────────────────────
''',
    )


# pylint: enable=line-too-long


def test_merge_swap_rzz_and_2q_unitaries_raises_if_tags_sames():
    with pytest.raises(ValueError, match="should be different"):
        sycamore_gateset.merge_swap_rzz_and_2q_unitaries(
            cirq.Circuit(),
            merged_swap_rzz_tag='merged_component',
            merged_2q_component_tag='merged_component',
        )


def test_merge_swap_rzz_and_2q_unitaries_deep():
    q = cirq.LineQubit.range(3)
    swap_rzz = cirq.FrozenCircuit(cirq.SWAP(*q[:2]), cirq.ZZ(*q[:2]) ** 0.5)
    rzz_swap = cirq.FrozenCircuit(cirq.ZZ(*q[1:]) ** 0.25, cirq.SWAP(*q[1:]))
    x_cnot_x = cirq.FrozenCircuit(cirq.X(q[0]), cirq.CNOT(*q[:2]), cirq.X(q[0]))
    x_cz_x = cirq.FrozenCircuit(cirq.X(q[2]), cirq.CZ(*q[1:]), cirq.X(q[2]))
    c_orig = cirq.Circuit(
        cirq.CircuitOperation(swap_rzz).repeat(3).with_tags("ignore"),
        cirq.CircuitOperation(rzz_swap).repeat(5).with_tags("preserve_tag"),
        cirq.CircuitOperation(x_cnot_x).repeat(7).with_tags("ignore"),
        cirq.CircuitOperation(x_cz_x).repeat(9).with_tags("preserve_tag"),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                [swap_rzz, rzz_swap, x_cnot_x, x_cz_x],
                cirq.Moment(cirq.Y(qq).with_tags("ignore") for qq in q),
            )
        ),
    )
    t_swap_rzz = "_merged_swap_rzz_tag"
    t_2q_cmp = "_merged_2q_unitaries_component"
    t_all = "_intermediate_result_tag_apply_to_all"

    def _wrap_cop(c: cirq.FrozenCircuit, *tags) -> cirq.FrozenCircuit:
        return cirq.FrozenCircuit(cirq.CircuitOperation(c).with_tags(*tags, t_all))

    c_expected = cirq.Circuit(
        cirq.CircuitOperation(swap_rzz).repeat(3).with_tags("ignore"),
        cirq.CircuitOperation(_wrap_cop(rzz_swap, t_swap_rzz)).repeat(5).with_tags("preserve_tag"),
        cirq.CircuitOperation(x_cnot_x).repeat(7).with_tags("ignore"),
        cirq.CircuitOperation(_wrap_cop(x_cz_x, t_2q_cmp)).repeat(9).with_tags("preserve_tag"),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                [_wrap_cop(swap_rzz, t_swap_rzz), _wrap_cop(rzz_swap, t_swap_rzz)],
                [_wrap_cop(x_cnot_x, t_2q_cmp), _wrap_cop(x_cz_x, t_2q_cmp)],
                cirq.Moment(cirq.Y(qq).with_tags("ignore") for qq in q),
            )
        ),
    )
    context = cirq.TransformerContext(tags_to_ignore=["ignore"], deep=True)
    c_new = sycamore_gateset.merge_swap_rzz_and_2q_unitaries(
        c_orig,
        context=context,
        merged_swap_rzz_tag=t_swap_rzz,
        merged_2q_component_tag=t_2q_cmp,
        intermediate_result_tag=t_all,
    )
    cirq.testing.assert_same_circuits(cirq.drop_empty_moments(c_new, context=context), c_expected)


def test_sycamore_gateset_compiles_swap_zz():
    qubits = cirq.LineQubit.range(3)

    gamma = np.random.randn()
    circuit1 = cirq.Circuit(
        cirq.SWAP(qubits[0], qubits[1]),
        cirq.Z(qubits[2]),
        cirq.ZZ(qubits[0], qubits[1]) ** gamma,
        strategy=cirq.InsertStrategy.NEW,
    )
    circuit2 = cirq.Circuit(
        cirq.ZZ(qubits[0], qubits[1]) ** gamma,
        cirq.Z(qubits[2]),
        cirq.SWAP(qubits[0], qubits[1]),
        strategy=cirq.InsertStrategy.NEW,
    )
    gateset = cirq_google.SycamoreTargetGateset()
    compiled_circuit1 = cirq.optimize_for_target_gateset(circuit1, gateset=gateset)
    compiled_circuit2 = cirq.optimize_for_target_gateset(circuit2, gateset=gateset)
    cirq.testing.assert_same_circuits(compiled_circuit1, compiled_circuit2)
    assert (
        len(list(compiled_circuit1.findall_operations_with_gate_type(cirq_google.SycamoreGate)))
        == 3
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit1, compiled_circuit1, atol=1e-7
    )


def test_convert_to_sycamore_gates_fsim():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.FSimGate(theta=np.pi / 2, phi=np.pi / 6)(q0, q1))
    compiled_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    cirq.testing.assert_same_circuits(circuit, compiled_circuit)


def test_single_qubit_gate():
    q = cirq.LineQubit(0)
    mat = cirq.testing.random_unitary(2)
    gate = cirq.MatrixGate(mat, qid_shape=(2,))
    circuit = cirq.Circuit(gate(q))
    compiled_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    ops = list(compiled_circuit.all_operations())
    assert len(ops) == 1
    assert isinstance(ops[0].gate, cirq.PhasedXZGate)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, compiled_circuit, atol=1e-8
    )


def test_single_qubit_gate_phased_xz():
    q = cirq.LineQubit(0)
    gate = cirq.PhasedXZGate(axis_phase_exponent=0.2, x_exponent=0.3, z_exponent=0.4)
    circuit = cirq.Circuit(gate(q))
    compiled_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    ops = list(compiled_circuit.all_operations())
    assert len(ops) == 1
    assert ops[0].gate == gate


def test_unsupported_gate():
    class UnknownGate(cirq.testing.TwoQubitGate):
        pass

    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(UnknownGate()(q0, q1))
    with pytest.raises(ValueError, match='Unable to convert'):
        cirq.optimize_for_target_gateset(
            circuit, gateset=cirq_google.SycamoreTargetGateset(), ignore_failures=False
        )


def test_nested_unsupported_gate():
    class UnknownGate(cirq.testing.TwoQubitGate):
        pass

    q0 = cirq.LineQubit(0)
    q1 = cirq.LineQubit(1)
    subcircuit = cirq.FrozenCircuit(UnknownGate()(q0, q1))
    circuit = cirq.Circuit(cirq.CircuitOperation(subcircuit))
    with pytest.raises(ValueError, match='Unable to convert'):
        cirq.optimize_for_target_gateset(
            circuit, gateset=cirq_google.SycamoreTargetGateset(), ignore_failures=False
        )


def test_unsupported_gate_ignoring_failures():
    class UnknownOperation(cirq.Operation):
        def __init__(self, qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            return self._qubits

        def with_qubits(self, *new_qubits):
            return UnknownOperation(self._qubits)  # pragma: no cover

    q0 = cirq.LineQubit(0)
    circuit = cirq.Circuit(UnknownOperation([q0]))
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    assert circuit == converted_circuit


def test_zztheta_qaoa_like():
    qubits = cirq.LineQubit.range(4)
    for exponent in np.linspace(-1, 1, 10):
        circuit = cirq.Circuit(
            [
                cirq.H.on_each(qubits),
                cirq.ZZPowGate(exponent=exponent)(qubits[0], qubits[1]),
                cirq.ZZPowGate(exponent=exponent)(qubits[2], qubits[3]),
                cirq.rx(0.123).on_each(qubits),
            ]
        )
        converted_circuit = cirq.optimize_for_target_gateset(
            circuit, gateset=cirq_google.SycamoreTargetGateset()
        )
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, converted_circuit, atol=1e-8
        )


def test_zztheta_zzpow_unsorted_qubits():
    qubits = cirq.LineQubit(1), cirq.LineQubit(0)
    exponent = 0.06366197723675814
    circuit = cirq.Circuit(
        cirq.ZZPowGate(exponent=exponent, global_shift=-0.5).on(qubits[0], qubits[1])
    )
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )


def test_swap_zztheta():
    qubits = cirq.LineQubit.range(2)
    a, b = qubits
    for theta in np.linspace(0, 2 * np.pi, 10):
        circuit = cirq.Circuit(
            cirq.SWAP(a, b), cirq.ZZPowGate(exponent=2 * theta / np.pi, global_shift=-0.5).on(a, b)
        )
        converted_circuit = cirq.optimize_for_target_gateset(
            circuit, gateset=cirq_google.SycamoreTargetGateset()
        )
        cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
            circuit, converted_circuit, atol=1e-8
        )


def test_known_two_q_operations_to_sycamore_operations_cnot():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.CNOT(a, b))
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )

    # Should be equivalent.
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )

    # Should have decomposed into two Sycamores.
    multi_qubit_ops = [e for e in converted_circuit.all_operations() if len(e.qubits) > 1]
    assert len(multi_qubit_ops) == 2
    assert all(isinstance(e.gate, cirq_google.SycamoreGate) for e in multi_qubit_ops)


@pytest.mark.parametrize(
    'gate',
    [
        cirq.MatrixGate(cirq.unitary(cirq.CX), qid_shape=(2, 2)),
        cirq.ISWAP,
        cirq.SWAP,
        cirq.CNOT,
        cirq.CZ,
        cirq.PhasedISwapPowGate(exponent=1.0),
        cirq.PhasedISwapPowGate(exponent=1.0, phase_exponent=0.33),
        cirq.PhasedISwapPowGate(exponent=0.66, phase_exponent=0.25),
        *[cirq.givens(theta) for theta in np.linspace(0, 2 * np.pi, 30)],
        *[cirq.ZZPowGate(exponent=2 * phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)],
        *[cirq.CZPowGate(exponent=phi / np.pi) for phi in np.linspace(0, 2 * np.pi, 30)],
    ],
)
def test_convert_to_sycamore_equivalent_unitaries(gate):
    circuit = cirq.Circuit(gate.on(*cirq.LineQubit.range(2)))
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )


def test_convert_to_sycamore_tabulation():
    # A tabulation for the sycamore gate with an infidelity of .1.
    sycamore_tabulation = cirq.two_qubit_gate_product_tabulation(
        cirq.unitary(cirq_google.SYC), 0.1, random_state=cirq.value.parse_random_state(11)
    )
    circuit = cirq.Circuit(cirq.MatrixGate(cirq.unitary(cirq.CX)).on(*cirq.LineQubit.range(2)))
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset(tabulation=sycamore_tabulation)
    )
    u1 = cirq.unitary(circuit)
    u2 = cirq.unitary(converted_circuit)
    overlap = abs(np.trace(u1.conj().T @ u2))
    assert np.isclose(overlap, 4.0, 0.1)


q = cirq.GridQubit.rect(1, 3)
matrix_gate = cirq.MatrixGate(cirq.testing.random_unitary(2))


@pytest.mark.parametrize(
    'op',
    [
        cirq.CircuitOperation(cirq.FrozenCircuit(matrix_gate(q[0]))),
        matrix_gate(q[0]),
        matrix_gate(q[0]).with_tags('test_tags'),
        matrix_gate(q[0]).controlled_by(q[1]),
        matrix_gate(q[0]).controlled_by(q[1]).with_tags('test_tags'),
        matrix_gate(q[0]).with_tags('test_tags').controlled_by(q[1]),
    ],
)
def test_supported_operation(op):
    circuit = cirq.Circuit(op)
    converted_circuit = cirq.optimize_for_target_gateset(
        circuit, gateset=cirq_google.SycamoreTargetGateset()
    )
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(
        circuit, converted_circuit, atol=1e-8
    )
    multi_qubit_ops = [e for e in converted_circuit.all_operations() if len(e.qubits) > 1]
    assert all(isinstance(e.gate, cirq_google.SycamoreGate) for e in multi_qubit_ops)


@pytest.mark.parametrize(
    'gateset',
    [
        cirq_google.SycamoreTargetGateset(),
        cirq_google.SycamoreTargetGateset(
            tabulation=cirq.two_qubit_gate_product_tabulation(
                cirq.unitary(cirq_google.SYC), 0.1, random_state=cirq.value.parse_random_state(11)
            )
        ),
    ],
)
def test_repr_json(gateset):
    assert eval(repr(gateset)) == gateset
    assert cirq.read_json(json_text=cirq.to_json(gateset)) == gateset
