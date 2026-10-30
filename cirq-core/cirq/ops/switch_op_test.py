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

import numpy as np
import pytest
import sympy

import cirq


class _AnyVersionQasmGate(cirq.Gate):
    """A single-qubit gate whose QASM output does not depend on the QASM version."""

    def _num_qubits_(self) -> int:
        return 1

    def _qasm_(self, args: cirq.QasmArgs, qubits: tuple[cirq.Qid, ...]) -> str:
        return args.format('g {0};\n', qubits[0])


class _NoQasmGate(cirq.Gate):
    def _num_qubits_(self) -> int:
        return 1


def test_init() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Switch('m', (cirq.Case((0,), cirq.X(q)), cirq.Case((1,), cirq.Y(q))))
    assert op.control == cirq.MeasurementKey('m')
    assert len(op.cases) == 2
    assert op.qubits == (q,)
    assert op.default is None


def test_init_case_values() -> None:
    q = cirq.LineQubit(0)
    sym = sympy.Symbol('s')
    with pytest.raises(ValueError, match='Case object needs at least one condition.'):
        _ = cirq.Case((), cirq.X(q))
    case_sym = cirq.Case((sym,), cirq.X(q))
    op = cirq.Switch('m', (case_sym,))
    assert op.cases[0].values == (sym,)


def test_init_multiple_operations() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.Switch('m', (cirq.Case((0,), (cirq.X(q0), cirq.Y(q1))),))
    assert isinstance(op.cases[0].then, cirq.CircuitOperation)
    assert op.cases[0].then.circuit == cirq.Circuit(cirq.X(q0), cirq.Y(q1))
    assert op.qubits == (q0, q1)


def test_init_errors() -> None:
    q = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='At least one case must be provided.'):
        _ = cirq.Switch('m', ())
    with pytest.raises(ValueError, match='At least one case must be provided.'):
        _ = cirq.Switch('m', (), default=cirq.X(q))


def test_with_qubits() -> None:
    q0, q1, q2, q3 = cirq.LineQubit.range(4)
    op = cirq.Switch('m', (cirq.Case((0,), cirq.X(q0)),), default=cirq.Y(q2))
    new_op = op.with_qubits(q1, q3)
    expected_op = cirq.Switch('m', (cirq.Case((0,), cirq.X(q1)),), default=cirq.Y(q3))
    assert new_op == expected_op


def test_with_qubits_wrong_count_raises() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    op = cirq.Switch('m', (cirq.Case((0,), cirq.X(q0)),), default=cirq.Y(q1))
    with pytest.raises(ValueError, match='Expected 2 qubits, got 1.'):
        _ = op.with_qubits(q2)
    with pytest.raises(ValueError, match='Expected 2 qubits, got 3.'):
        _ = op.with_qubits(q0, q1, q2)


def test_without_classical_controls_single_branch() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Switch('m', (cirq.Case((0,), cirq.X(q)),))
    assert op.without_classical_controls() == cirq.X(q)


def test_without_classical_controls_multiple_branches() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    op = cirq.Switch(
        'm',
        (cirq.Case((0,), cirq.X(q0)), cirq.Case((1,), (cirq.Y(q0), cirq.Z(q1)))),
        default=cirq.H(q1),
    )
    assert op.without_classical_controls() == cirq.CircuitOperation(
        cirq.FrozenCircuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q1), cirq.H(q1))
    )


def test_value_equality() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.Switch('a', (cirq.Case((0,), cirq.X(q0)),)),
        cirq.Switch(cirq.MeasurementKey('a'), (cirq.Case((0,), cirq.X(q0)),)),
    )
    eq.add_equality_group(cirq.Switch('b', (cirq.Case((0,), cirq.X(q0)),)))
    eq.add_equality_group(cirq.Switch('a', (cirq.Case((1,), cirq.X(q0)),)))
    eq.add_equality_group(cirq.Switch('a', (cirq.Case((0,), cirq.X(q0)),), default=cirq.Y(q0)))


def test_str_and_repr() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Switch(
        'm', (cirq.Case((0,), cirq.X(q)), cirq.Case((1,), cirq.Y(q))), default=cirq.Z(q)
    )
    assert str(op) is not None
    assert repr(op) is not None
    assert eval(repr(op)) == op


def test_parameterized_and_resolve() -> None:
    q = cirq.LineQubit(0)
    sym = sympy.Symbol('theta')
    case_sym = sympy.Symbol('val')

    op = cirq.Switch('m', (cirq.Case((case_sym,), cirq.Rx(rads=sym).on(q)),))
    assert cirq.is_parameterized(op)
    assert 'theta' in cirq.parameter_names(op)

    resolved = cirq.resolve_parameters(op, cirq.ParamResolver({'theta': np.pi, 'val': 0}))
    assert not cirq.is_parameterized(resolved)
    assert resolved == cirq.Switch('m', (cirq.Case((0,), cirq.Rx(rads=np.pi).on(q)),))


def test_parameter_names_with_default() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Switch(
        'm',
        (cirq.Case((0,), cirq.Rx(rads=sympy.Symbol('a')).on(q)),),
        default=cirq.Ry(rads=sympy.Symbol('b')).on(q),
    )
    assert cirq.parameter_names(op) == {'a', 'b'}


def test_diagram() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='m'),
        cirq.Switch('m', (cirq.Case((0,), cirq.X(q1)), cirq.Case((1,), cirq.Y(q1)))),
    )
    diagram = circuit.to_text_diagram(use_unicode_characters=True)
    assert 'Switch' in diagram


def test_simulation() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    # Measure q0 as 0, should trigger case (0,) flipping q1.
    circuit = cirq.Circuit(
        cirq.measure(q0, key='m'),
        cirq.Switch('m', (cirq.Case((0,), cirq.X(q1)), cirq.Case((1,), cirq.Y(q1)))),
        cirq.measure(q1, key='res'),
    )
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    assert res.measurements['m'] == [0]
    assert res.measurements['res'] == [1]


def test_simulation_run_default() -> None:
    q0, q1, q2 = cirq.LineQubit.range(3)
    # Measure q0 as 0, should trigger case (0,) flipping q1.
    circuit = cirq.Circuit(
        cirq.measure(q0, key='m'),
        cirq.Switch(
            'm', (cirq.Case((0,), cirq.X(q1)), cirq.Case((1,), cirq.Y(q1))), default=cirq.X(q2)
        ),
        cirq.measure(q1, key='res'),
        cirq.measure(q2, key='def'),
    )
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    assert res.measurements['m'] == [0]
    assert res.measurements['res'] == [1]
    assert res.measurements['def'] == [0]

    circuit = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='m'),
        cirq.Switch('m', (cirq.Case((0,), cirq.X(q1)),), default=cirq.X(q2)),
        cirq.measure(q1, key='res'),
        cirq.measure(q2, key='def'),
    )
    sim = cirq.Simulator()
    res = sim.simulate(circuit)
    assert res.measurements['m'] == [1]
    assert res.measurements['res'] == [0]
    assert res.measurements['def'] == [1]


def test_qasm() -> None:
    q1 = cirq.LineQubit(1)
    op = cirq.Switch('a', (cirq.Case((0,), cirq.X(q1)), cirq.Case((1,), cirq.Y(q1))))
    # Note that `switch` is only available after 3.1 but at the time of creating this test, no
    # operation's _qasm_ function explicitly supports 3.1. To actually evaluate this test, we will
    # just use 3.0 temporarily.
    args = cirq.QasmArgs(version='3.0', qubit_id_map={q1: 'q[1]'}, meas_key_id_map={'a': 'm_a'})
    with pytest.raises(ValueError, match='QASM version 3.0 output is not supported'):
        _ = cirq.qasm(op, args=args)
    assert op.qasm(args=args) == (
        'switch (m_a) {\n'
        '    case 0 {\n'
        '        x q[1];\n'
        '    }\n'
        '    case 1 {\n'
        '        y q[1];\n'
        '    }\n'
        '}'
    )


def test_qasm_with_default() -> None:
    q1 = cirq.LineQubit(1)
    op = cirq.Switch('a', (cirq.Case((0, 2), cirq.X(q1)),), default=cirq.Y(q1))
    args = cirq.QasmArgs(version='3.0', qubit_id_map={q1: 'q[1]'}, meas_key_id_map={'a': 'm_a'})
    assert op.qasm(args=args) == (
        'switch (m_a) {\n'
        '    case 0, 2 {\n'
        '        x q[1];\n'
        '    }\n'
        '    default {\n'
        '        y q[1];\n'
        '    }\n'
        '}'
    )


def test_qasm_default_without_qasm() -> None:
    q1 = cirq.LineQubit(1)
    op = cirq.Switch('a', (cirq.Case((0,), cirq.X(q1)),), default=_NoQasmGate().on(q1))
    args = cirq.QasmArgs(version='3.0', qubit_id_map={q1: 'q[1]'}, meas_key_id_map={'a': 'm_a'})
    assert op.qasm(args=args) is None


def test_qasm_protocol_v3_1() -> None:
    q1 = cirq.LineQubit(1)
    gate = _AnyVersionQasmGate()
    op = cirq.Switch('a', (cirq.Case((0,), gate(q1)),), default=gate(q1))
    args = cirq.QasmArgs(version='3.1', qubit_id_map={q1: 'q[1]'}, meas_key_id_map={'a': 'm_a'})
    assert cirq.qasm(op, args=args) == (
        'switch (m_a) {\n'
        '    case 0 {\n'
        '        g q[1];\n'
        '    }\n'
        '    default {\n'
        '        g q[1];\n'
        '    }\n'
        '}'
    )


def test_qasm_protocol_default_args_rejects_version() -> None:
    q = cirq.LineQubit(0)
    op = cirq.Switch('a', (cirq.Case((0,), cirq.X(q)),))
    # The default `cirq.QasmArgs` uses QASM 2.0, which does not support `switch`.
    with pytest.raises(ValueError, match='QASM version 2.0 output is not supported'):
        _ = op._qasm_()


def test_case_qasm_protocol_v3_1() -> None:
    q1 = cirq.LineQubit(1)
    case = cirq.Case((0, 1), _AnyVersionQasmGate().on(q1))
    args = cirq.QasmArgs(version='3.1', qubit_id_map={q1: 'q[1]'})
    assert cirq.qasm(case, args=args) == 'case 0, 1 {\n    g q[1];\n}'


def test_case_qasm_protocol_default_args_rejects_version() -> None:
    case = cirq.Case((0,), cirq.X(cirq.LineQubit(0)))
    with pytest.raises(ValueError, match='QASM version 2.0 output is not supported'):
        _ = case._qasm_()
