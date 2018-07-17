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

import pytest

import cirq


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def test_empty_circuit():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((), (q0,))
    assert (str(output) ==
"""OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
""")


def test_header():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((), (q0,), header=
"""My test circuit
Device: Bristlecone""")
    assert (str(output) ==
"""// My test circuit
// Device: Bristlecone

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
""")

    output = cirq.QasmOutput((), (q0,), header=
"""
My test circuit
Device: Bristlecone
""")
    assert (str(output) ==
"""//
// My test circuit
// Device: Bristlecone
//

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];
""")


def test_single_gate_no_parameter():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0),), (q0,))
    assert (str(output) ==
"""OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
""")


def test_single_gate_with_parameter():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0) ** 0.25,), (q0,))
    assert (str(output) ==
"""OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


rx(pi*0.25) q[0];
""")


def test_precision():
    q0, = _make_qubits(1)
    output = cirq.QasmOutput((cirq.X(q0) ** 0.1234567,), (q0,), precision=3)
    assert (str(output) ==
"""OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


rx(pi*0.123) q[0];
""")


def test_version():
    q0, = _make_qubits(1)
    with pytest.raises(ValueError):
        output = cirq.QasmOutput((), (q0,), version='3.0')
        _ = str(output)


def test_everything():
    q0, q1, q2, q3, q4 = _make_qubits(5)
    circuit = cirq.Circuit.from_ops(
        cirq.Z(q0),
        cirq.Z(q0) ** .1,
        cirq.Y(q0),
        cirq.Y(q0) ** .2,
        cirq.X(q0),
        cirq.X(q0) ** .333333333333333333333333333,
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.CZ(q0, q1) ** .4,

        cirq.CCZ(q0, q1, q2),
        cirq.CCX(q0, q1, q2),
        cirq.CSWAP(q0, q1, q2),

        cirq.ISWAP(q2, q0),

        cirq.google.ExpZGate()(q3),
        cirq.google.ExpZGate(half_turns=0.5)(q3),
        cirq.google.ExpWGate(axis_half_turns=.6, half_turns=.7)(q1),

        cirq.MeasurementGate('xX')(q0),
        cirq.MeasurementGate('x_a')(q2),
        cirq.MeasurementGate('x?')(q1),
        cirq.MeasurementGate('X')(q3),
        cirq.MeasurementGate('_x')(q4),
        cirq.MeasurementGate('multi', (False, True))(q1, q2, q3),
    )
    output = cirq.QasmOutput(circuit.all_operations(), (q0, q1, q2, q3, q4),
                             header='Generated from Cirq')
    assert (str(output) ==
"""// Generated from Cirq

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0, q1, q2, q3, q4]
qreg q[5];
creg m_xX[1];
creg m_x_a[1];
creg m0[x?];  // Measurement: 1
creg m_X[1];
creg m__x[1];
creg m_multi[3];


z q[0];
rz(pi*0.1) q[0];
y q[0];
ry(pi*0.2) q[0];
x q[0];
rx(pi*0.3333333333) q[0];
h q[1];
cz q[0],q[1];

// Gate: CZ**0.3999999999999999
u3(pi*0.5,pi*1.0,pi*0.75) q[0];
u3(pi*0.5,pi*1.0,pi*1.25) q[1];
rx(pi*0.5) q[0];
cx q[0],q[1];
rx(pi*0.3) q[0];
ry(pi*0.5) q[1];
cx q[1],q[0];
rx(pi*-0.5) q[1];
rz(pi*0.5) q[1];
cx q[0],q[1];
u3(pi*0.5,pi*0.45,0) q[0];
u3(pi*0.5,pi*1.95,0) q[1];

h q[2];
ccx q[0],q[1],q[2];
h q[2];
ccx q[0],q[1],q[2];
cswap q[0],q[1],q[2];

// Gate: ISWAP
cx q[2],q[0];
h q[2];
cx q[0],q[2];
rz(pi*0.5) q[2];
cx q[0],q[2];
rz(pi*-0.5) q[2];
h q[2];
cx q[2],q[0];

z q[3];
rz(pi*0.5) q[3];

// Gate: W(0.6)^0.7
u3(pi*0.7,pi*0.1,pi*1.9) q[1];

measure q[0] -> m_xX[0];
measure q[2] -> m_x_a[0];
measure q[1] -> m0[0];
measure q[3] -> m_X[0];
measure q[4] -> m__x[0];
measure q[1] -> m_multi[0];
x q[2];  // Invert the following measurement
measure q[2] -> m_multi[1];
measure q[3] -> m_multi[2];
""")
