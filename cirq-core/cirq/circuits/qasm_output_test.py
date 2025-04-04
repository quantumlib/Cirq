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
import os
import re

import numpy as np
import pytest

import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq


def _make_qubits(n):
    return [cirq.NamedQubit(f'q{i}') for i in range(n)]


def test_u_gate_repr():
    gate = QasmUGate(0.1, 0.2, 0.3)
    assert repr(gate) == 'cirq.circuits.qasm_output.QasmUGate(theta=0.1, phi=0.2, lmda=0.3)'


def test_u_gate_eq():
    gate = QasmUGate(0.1, 0.2, 0.3)
    gate2 = QasmUGate(0.1, 0.2, 0.3)
    cirq.approx_eq(gate, gate2, atol=1e-16)
    gate3 = QasmUGate(0.1, 0.2, 0.4)
    gate4 = QasmUGate(0.1, 0.2, 2.4)
    cirq.approx_eq(gate4, gate3, atol=1e-16)


def test_qasm_two_qubit_gate_repr():
    cirq.testing.assert_equivalent_repr(
        QasmTwoQubitGate.from_matrix(cirq.testing.random_unitary(4))
    )


def test_qasm_u_qubit_gate_unitary():
    u = cirq.testing.random_unitary(2)
    g = QasmUGate.from_matrix(u)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(g), u, atol=1e-7)

    cirq.testing.assert_implements_consistent_protocols(g)

    u = cirq.unitary(cirq.Y)
    g = QasmUGate.from_matrix(u)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(g), u, atol=1e-7)
    cirq.testing.assert_implements_consistent_protocols(g)


def test_qasm_two_qubit_gate_unitary():
    u = cirq.testing.random_unitary(4)
    g = QasmTwoQubitGate.from_matrix(u)
    np.testing.assert_allclose(cirq.unitary(g), u)


def test_empty_circuit_one_qubit():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((), (q0,))
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
"""
    )


def test_empty_circuit_no_qubits():
    output = cirq.QasmOutput((), ())
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: []
"""
    )


def test_header():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput(
        (),
        (q0,),
        header="""My test circuit
Device: Bristlecone""",
    )
    assert (
        str(output)
        == """// My test circuit
// Device: Bristlecone

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
"""
    )

    output = cirq.QasmOutput(
        (),
        (q0,),
        header="""
My test circuit
Device: Bristlecone
""",
    )
    assert (
        str(output)
        == """//
// My test circuit
// Device: Bristlecone
//

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
"""
    )


def test_single_gate_no_parameter():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0),), (q0,))
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
"""
    )


def test_single_gate_with_parameter():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0) ** 0.25,), (q0,))
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


rx(pi*0.25) q[0];
"""
    )


def test_h_gate_with_parameter():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((cirq.H(q0) ** 0.25,), (q0,))
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


// Gate: H**0.25
ry(pi*0.25) q[0];
rx(pi*0.25) q[0];
ry(pi*-0.25) q[0];
"""
    )


def test_qasm_global_pahse():
    output = cirq.QasmOutput((cirq.global_phase_operation(np.exp(1j * 5))), ())
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: []
"""
    )


def test_precision():
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0) ** 0.1234567,), (q0,), precision=3)
    assert (
        str(output)
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


rx(pi*0.123) q[0];
"""
    )


def test_version():
    (q0,) = _make_qubits(1)
    with pytest.raises(ValueError):
        output = cirq.QasmOutput((), (q0,), version='4.0')
        _ = str(output)


def test_save_to_file(tmpdir):
    file_path = os.path.join(tmpdir, 'test.qasm')
    (q0,) = _make_qubits(1)
    output = cirq.QasmOutput((), (q0,))
    output.save(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
    assert (
        file_content
        == """OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
"""
    )


def test_unsupported_operation():
    (q0,) = _make_qubits(1)

    class UnsupportedOperation(cirq.Operation):
        qubits = (q0,)
        with_qubits = NotImplemented

    output = cirq.QasmOutput((UnsupportedOperation(),), (q0,))
    with pytest.raises(ValueError):
        _ = str(output)


def _all_operations(q0, q1, q2, q3, q4, include_measurements=True):
    class ExampleOperation(cirq.Operation):
        qubits = (q0,)
        with_qubits = NotImplemented

        def _qasm_(self, args: cirq.QasmArgs) -> str:
            return '// Example operation\n'

        def _decompose_(self):
            # Only used by test_output_unitary_same_as_qiskit
            return ()  # pragma: no cover

    class ExampleCompositeOperation(cirq.Operation):
        qubits = (q0,)
        with_qubits = NotImplemented

        def _decompose_(self):
            return cirq.X(self.qubits[0])

        def __repr__(self):
            return 'ExampleCompositeOperation()'

    return (
        cirq.I(q0),
        cirq.Z(q0),
        cirq.Z(q0) ** 0.625,
        cirq.Z(q0) ** 0,
        cirq.Y(q0),
        cirq.Y(q0) ** 0.375,
        cirq.Y(q0) ** 0,
        cirq.X(q0),
        cirq.X(q0) ** 0.875,
        cirq.X(q0) ** 0,
        cirq.H(q0),
        cirq.H(q0) ** 0,
        cirq.X(q0) ** 0.5,
        cirq.X(q0) ** -0.5,
        cirq.S(q0),
        cirq.Z(q0) ** -0.5,
        cirq.T(q0),
        cirq.Z(q0) ** -0.25,
        cirq.Rx(rads=np.pi)(q0),
        cirq.Rx(rads=np.pi / 2)(q0),
        cirq.Rx(rads=np.pi / 4)(q0),
        cirq.Ry(rads=np.pi)(q0),
        cirq.Ry(rads=np.pi / 2)(q0),
        cirq.Ry(rads=np.pi / 4)(q0),
        cirq.Rz(rads=np.pi)(q0),
        cirq.Rz(rads=np.pi / 2)(q0),
        cirq.Rz(rads=np.pi / 4)(q0),
        cirq.MatrixGate(cirq.unitary(cirq.H) @ cirq.unitary(cirq.T)).on(q0),
        cirq.CZ(q0, q1),
        cirq.CZ(q0, q1) ** 0.25,  # Requires 2-qubit decomposition
        cirq.CNOT(q0, q1),
        cirq.CNOT(q0, q1) ** 0.5,  # Requires 2-qubit decomposition
        cirq.ControlledGate(cirq.Y)(q0, q1),
        cirq.ControlledGate(cirq.H)(q0, q1),
        cirq.SWAP(q0, q1),
        cirq.SWAP(q0, q1) ** 0.75,  # Requires 2-qubit decomposition
        cirq.CCZ(q0, q1, q2),
        cirq.CCX(q0, q1, q2),
        cirq.CCZ(q0, q1, q2) ** 0.5,
        cirq.CCX(q0, q1, q2) ** 0.5,
        cirq.CSWAP(q0, q1, q2),
        cirq.IdentityGate(1).on(q0),
        cirq.IdentityGate(3).on(q0, q1, q2),
        cirq.ISWAP(q2, q0),  # Requires 2-qubit decomposition
        cirq.PhasedXPowGate(phase_exponent=0.111, exponent=0.25).on(q1),
        cirq.PhasedXPowGate(phase_exponent=0.333, exponent=0.5).on(q1),
        cirq.PhasedXPowGate(phase_exponent=0.777, exponent=-0.5).on(q1),
        (
            (
                cirq.measure(q0, key='xX'),
                cirq.measure(q2, key='x_a'),
                cirq.measure(q1, key='x?'),
                cirq.measure(q3, key='X'),
                cirq.measure(q4, key='_x'),
                cirq.measure(q2, key='x_a'),
                cirq.measure(q1, q2, q3, key='multi', invert_mask=(False, True)),
            )
            if include_measurements
            else ()
        ),
        ExampleOperation(),
        ExampleCompositeOperation(),
    )


def test_output_unitary_same_as_qiskit():
    qubits = tuple(_make_qubits(5))
    operations = _all_operations(*qubits, include_measurements=False)
    output = cirq.QasmOutput(operations, qubits, header='Generated from Cirq', precision=10)
    text = str(output)

    circuit = cirq.Circuit(operations)
    cirq_unitary = circuit.unitary(qubit_order=qubits)
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(text, cirq_unitary)


def test_fails_on_big_unknowns():
    class UnrecognizedGate(cirq.testing.ThreeQubitGate):
        pass

    c = cirq.Circuit(UnrecognizedGate().on(*cirq.LineQubit.range(3)))
    with pytest.raises(ValueError, match='Cannot output operation as QASM'):
        _ = c.to_qasm()


def test_output_format():
    def filter_unpredictable_numbers(text):
        return re.sub(r'u3\(.+\)', r'u3(<not-repeatable>)', text)

    qubits = tuple(_make_qubits(5))
    operations = _all_operations(*qubits)
    output = cirq.QasmOutput(operations, qubits, header='Generated from Cirq!', precision=5)
    assert filter_unpredictable_numbers(str(output)) == filter_unpredictable_numbers(
        """// Generated from Cirq!

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0, q1, q2, q3, q4]
qreg q[5];
creg m_xX[1];
creg m_x_a[1];
creg m0[1];  // Measurement: x?
creg m_X[1];
creg m__x[1];
creg m_multi[3];


id q[0];
z q[0];
rz(pi*0.625) q[0];
rz(0) q[0];
y q[0];
ry(pi*0.375) q[0];
ry(0) q[0];
x q[0];
rx(pi*0.875) q[0];
rx(0) q[0];
h q[0];
id q[0];
sx q[0];
sxdg q[0];
s q[0];
sdg q[0];
t q[0];
tdg q[0];
rx(pi*1.0) q[0];
rx(pi*0.5) q[0];
rx(pi*0.25) q[0];
ry(pi*1.0) q[0];
ry(pi*0.5) q[0];
ry(pi*0.25) q[0];
rz(pi*1.0) q[0];
rz(pi*0.5) q[0];
rz(pi*0.25) q[0];
u3(pi*1.5,pi*1.0,pi*0.25) q[0];
cz q[0],q[1];

// Gate: CZ**0.25
u3(pi*0.5,pi*1.0,pi*0.75) q[0];
u3(pi*0.5,pi*1.0,pi*0.25) q[1];
sx q[0];
cx q[0],q[1];
rx(pi*0.375) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(pi*0.5,pi*0.375,0) q[0];
u3(pi*0.5,pi*0.875,0) q[1];

cx q[0],q[1];

// Gate: CNOT**0.5
ry(pi*-0.5) q[1];
u3(pi*0.5,0,pi*0.25) q[0];
u3(pi*0.5,0,pi*0.75) q[1];
sx q[0];
cx q[0],q[1];
rx(pi*0.25) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
sxdg q[1];
s q[1];
cx q[0],q[1];
u3(pi*0.5,pi*1.0,pi*1.0) q[0];
u3(pi*0.5,pi*0.5,pi*1.0) q[1];
ry(pi*0.5) q[1];

cy q[0],q[1];
ch q[0],q[1];
swap q[0],q[1];

// Gate: SWAP**0.75
cx q[0],q[1];
ry(pi*-0.5) q[0];
u3(pi*0.5,0,pi*0.45919) q[1];
u3(pi*0.5,0,pi*1.95919) q[0];
sx q[1];
cx q[1],q[0];
rx(pi*0.125) q[1];
ry(pi*0.5) q[0];
cx q[0],q[1];
sxdg q[0];
s q[0];
cx q[1],q[0];
u3(pi*0.5,pi*0.91581,pi*1.0) q[1];
u3(pi*0.5,pi*1.41581,pi*1.0) q[0];
ry(pi*0.5) q[0];
cx q[0],q[1];

// Gate: CCZ
h q[2];
ccx q[0],q[1],q[2];
h q[2];

ccx q[0],q[1],q[2];

// Gate: CCZ**0.5
rz(pi*0.125) q[0];
rz(pi*0.125) q[1];
rz(pi*0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[1];
rz(pi*0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];

// Gate: TOFFOLI**0.5
h q[2];
rz(pi*0.125) q[0];
rz(pi*0.125) q[1];
rz(pi*0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[1];
rz(pi*0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
rz(pi*-0.125) q[2];
cx q[0],q[1];
cx q[1],q[2];
h q[2];

cswap q[0],q[1],q[2];
id q[0];

// Gate: I(3)
id q[0];
id q[1];
id q[2];

// Gate: ISWAP
cx q[2],q[0];
h q[2];
cx q[0],q[2];
s q[2];
cx q[0],q[2];
sdg q[2];
h q[2];
cx q[2],q[0];

u3(pi*-0.25, pi*0.611, pi*-0.611) q[1];
u2(pi*-0.167, pi*0.167) q[1];
u2(pi*1.277, pi*-1.277) q[1];
measure q[0] -> m_xX[0];
measure q[2] -> m_x_a[0];
measure q[1] -> m0[0];
measure q[3] -> m_X[0];
measure q[4] -> m__x[0];
measure q[2] -> m_x_a[0];

// Gate: cirq.MeasurementGate(3, cirq.MeasurementKey(name='multi'), (False, True))
measure q[1] -> m_multi[0];
x q[2];  // Invert the following measurement
measure q[2] -> m_multi[1];
x q[2];  // Undo the inversion
measure q[3] -> m_multi[2];

// Example operation

// Operation: ExampleCompositeOperation()
x q[0];
"""
    )


def test_reset():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.reset(a), cirq.reset(b))
    output = cirq.QasmOutput(
        c.all_operations(),
        tuple(sorted(c.all_qubits())),
        header='Generated from Cirq!',
        precision=5,
    )
    assert (
        str(output).strip()
        == """
// Generated from Cirq!

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1)]
qreg q[2];


h q[0];
cx q[0],q[1];
reset q[0];
reset q[1];
    """.strip()
    )


def test_different_sized_registers():
    qubits = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(qubits[0], key='c'), cirq.measure(qubits, key='c'))
    output = cirq.QasmOutput(
        c.all_operations(), tuple(sorted(c.all_qubits())), header='Generated from Cirq!'
    )
    assert (
        str(output)
        == """// Generated from Cirq!

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q(0), q(1)]
qreg q[2];
creg m_c[2];


measure q[0] -> m_c[0];

// Gate: cirq.MeasurementGate(2, cirq.MeasurementKey(name='c'), ())
measure q[0] -> m_c[0];
measure q[1] -> m_c[1];
"""
    )
    # OPENQASM 3.0
    output3 = cirq.QasmOutput(
        c.all_operations(),
        tuple(sorted(c.all_qubits())),
        header='Generated from Cirq!',
        version='3.0',
    )
    assert (
        str(output3)
        == """// Generated from Cirq!

OPENQASM 3.0;
include "stdgates.inc";


// Qubits: [q(0), q(1)]
qubit[2] q;
bit[2] m_c;


m_c[0] = measure q[0];

// Gate: cirq.MeasurementGate(2, cirq.MeasurementKey(name='c'), ())
m_c[0] = measure q[0];
m_c[1] = measure q[1];
"""
    )
