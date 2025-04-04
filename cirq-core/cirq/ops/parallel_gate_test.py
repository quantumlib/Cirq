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

import numpy as np
import pytest
import sympy

import cirq


@pytest.mark.parametrize(
    'gate, num_copies, qubits',
    [
        (cirq.testing.SingleQubitGate(), 2, cirq.LineQubit.range(2)),
        (cirq.X**0.5, 4, cirq.LineQubit.range(4)),
    ],
)
def test_parallel_gate_operation_init(gate, num_copies, qubits):
    v = cirq.ParallelGate(gate, num_copies)
    assert v.sub_gate == gate
    assert v.num_copies == num_copies
    assert v.on(*qubits).qubits == tuple(qubits)


@pytest.mark.parametrize(
    'gate, num_copies, qubits, error_msg',
    [
        (cirq.testing.SingleQubitGate(), 3, cirq.LineQubit.range(2), "Wrong number of qubits"),
        (
            cirq.testing.SingleQubitGate(),
            0,
            cirq.LineQubit.range(4),
            "gate must be applied at least once",
        ),
        (
            cirq.testing.SingleQubitGate(),
            2,
            [cirq.NamedQubit("a"), cirq.NamedQubit("a")],
            "Duplicate",
        ),
        (cirq.testing.TwoQubitGate(), 2, cirq.LineQubit.range(4), "must be a single qubit gate"),
    ],
)
def test_invalid_parallel_gate_operation(gate, num_copies, qubits, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        cirq.ParallelGate(gate, num_copies)(*qubits)


@pytest.mark.parametrize(
    'gate, num_copies, qubits',
    [(cirq.X, 2, cirq.LineQubit.range(2)), (cirq.H**0.5, 4, cirq.LineQubit.range(4))],
)
def test_decompose(gate, num_copies, qubits):
    g = cirq.ParallelGate(gate, num_copies)
    step = gate.num_qubits()
    qubit_lists = [qubits[i * step : (i + 1) * step] for i in range(num_copies)]
    assert set(cirq.decompose_once(g(*qubits))) == set(gate.on_each(qubit_lists))


def test_decompose_raises():
    g = cirq.ParallelGate(cirq.X, 2)
    qubits = cirq.LineQubit.range(4)
    with pytest.raises(ValueError, match=r'len\(qubits\)=4 should be 2'):
        cirq.decompose_once_with_qubits(g, qubits)


def test_with_num_copies():
    g = cirq.testing.SingleQubitGate()
    pg = cirq.ParallelGate(g, 3)
    assert pg.with_num_copies(5) == cirq.ParallelGate(g, 5)


def test_extrapolate():
    # If the gate isn't extrapolatable, you get a type error.
    g = cirq.ParallelGate(cirq.testing.SingleQubitGate(), 2)
    with pytest.raises(TypeError):
        _ = g**0.5
    # If the gate is extrapolatable, the effect is applied on the underlying gate.
    g = cirq.ParallelGate(cirq.Y, 2)
    assert g**0.5 == cirq.ParallelGate(cirq.Y**0.5, 2)
    assert cirq.inverse(g) == g**-1 == cirq.ParallelGate(cirq.Y**-1, 2)


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameterizable_gates(resolve_fn):
    r = cirq.ParamResolver({'a': 0.5})
    g1 = cirq.ParallelGate(cirq.Z ** sympy.Symbol('a'), 2)
    assert cirq.is_parameterized(g1)
    g2 = resolve_fn(g1, r)
    assert not cirq.is_parameterized(g2)


@pytest.mark.parametrize('gate', [cirq.X ** sympy.Symbol("a"), cirq.testing.SingleQubitGate()])
def test_no_unitary(gate):
    g = cirq.ParallelGate(gate, 2)
    assert not cirq.has_unitary(g)
    assert cirq.unitary(g, None) is None


@pytest.mark.parametrize(
    'gate, num_copies, qubits',
    [
        (cirq.X**0.5, 2, cirq.LineQubit.range(2)),
        (cirq.MatrixGate(cirq.unitary(cirq.H**0.25)), 6, cirq.LineQubit.range(6)),
    ],
)
def test_unitary(gate, num_copies, qubits):
    g = cirq.ParallelGate(gate, num_copies)
    step = gate.num_qubits()
    qubit_lists = [qubits[i * step : (i + 1) * step] for i in range(num_copies)]
    np.testing.assert_allclose(
        cirq.unitary(g), cirq.unitary(cirq.Circuit(gate.on_each(qubit_lists))), atol=1e-8
    )


def test_not_implemented_diagram():
    q = cirq.LineQubit.range(2)
    g = cirq.testing.SingleQubitGate()
    c = cirq.Circuit()
    c.append(cirq.ParallelGate(g, 2)(*q))
    assert 'cirq.testing.gate_features.SingleQubitGate ' in str(c)


def test_repr():
    assert repr(cirq.ParallelGate(cirq.X, 2)) == 'cirq.ParallelGate(sub_gate=cirq.X, num_copies=2)'


def test_str():
    assert str(cirq.ParallelGate(cirq.X**0.5, 10)) == 'X**0.5 x 10'


def test_equivalent_circuit():
    qreg = cirq.LineQubit.range(4)
    oldc = cirq.Circuit()
    newc = cirq.Circuit()
    single_qubit_gates = [cirq.X ** (1 / 2), cirq.Y ** (1 / 3), cirq.Z**-1]
    for gate in single_qubit_gates:
        for qubit in qreg:
            oldc.append(gate.on(qubit))
        newc.append(cirq.ParallelGate(gate, 4)(*qreg))
    cirq.testing.assert_has_diagram(newc, oldc.to_text_diagram())
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(oldc, newc, atol=1e-6)


@pytest.mark.parametrize('gate, num_copies', [(cirq.X, 1), (cirq.Y, 2), (cirq.Z, 3), (cirq.H, 4)])
def test_parallel_gate_operation_is_consistent(gate, num_copies):
    cirq.testing.assert_implements_consistent_protocols(cirq.ParallelGate(gate, num_copies))


def test_trace_distance():
    s = cirq.X**0.25
    two_g = cirq.ParallelGate(s, 2)
    three_g = cirq.ParallelGate(s, 3)
    four_g = cirq.ParallelGate(s, 4)
    assert cirq.approx_eq(cirq.trace_distance_bound(two_g), np.sin(np.pi / 4))
    assert cirq.approx_eq(cirq.trace_distance_bound(three_g), np.sin(3 * np.pi / 8))
    assert cirq.approx_eq(cirq.trace_distance_bound(four_g), 1.0)
    spg = cirq.ParallelGate(cirq.X ** sympy.Symbol('a'), 4)
    assert cirq.approx_eq(cirq.trace_distance_bound(spg), 1.0)


@pytest.mark.parametrize('gate, num_copies', [(cirq.X, 1), (cirq.Y, 2), (cirq.Z, 3), (cirq.H, 4)])
def test_parallel_gate_op(gate, num_copies):
    qubits = cirq.LineQubit.range(num_copies * gate.num_qubits())
    assert cirq.parallel_gate_op(gate, *qubits) == cirq.ParallelGate(gate, num_copies).on(*qubits)
