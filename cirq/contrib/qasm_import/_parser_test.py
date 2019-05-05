#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest
import sympy
from sympy import Number
import numpy as np
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser


def test_format_header_circuit():
    parser = QasmParser("OPENQASM 2.0;")

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert not parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


def test_unsupported_format():
    qasm = "OPENQASM 2.1;"
    parser = QasmParser(qasm)

    try:
        parser.parse()
        raise AssertionError("should fail with no format error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "Unsupported OpenQASM version: 2.1, " \
                             "only 2.0 is supported currently by Cirq"


def test_format_header_with_quelibinc_circuit():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
"""
    parser = QasmParser(qasm)

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


def test_error_not_starting_with_format():
    qasm = "include \"qelib1.inc\";"
    parser = QasmParser(qasm)
    try:
        parser.parse()
        raise AssertionError("should fail with no format error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "Missing 'OPENQASM 2.0;' statement"


def test_error_on_empty():
    parser = QasmParser("")
    try:
        parser.parse()
        raise AssertionError("should fail with no format error")
    except QasmException as ex:
        assert ex.message == "Unexpected end of file"


def test_multiple_qreg_declaration():
    qasm = """
     OPENQASM 2.0; 
     include "qelib1.inc";
     qreg a_quantum_register [ 1337 ];
     qreg q[42];
"""
    parser = QasmParser(qasm)

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())
    assert parsed_qasm.qregs == {'a_quantum_register': 1337, 'q': 42}


def test_multiple_creg_declaration():
    qasm = """
     OPENQASM 2.0; 
     include "qelib1.inc";
     creg a_classical_register [1337];
     qreg a_quantum_register [1337];
     creg c[42];
"""
    parser = QasmParser(qasm)

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())
    assert parsed_qasm.qregs == {'a_quantum_register': 1337}
    assert parsed_qasm.cregs == {'a_classical_register': 1337, 'c': 42}


def test_CX_gate():
    qasm = """
     OPENQASM 2.0;          
     qreg q1[2];
     qreg q2[2];
     CX q1[0], q1[1];
     CX q1, q2[0];
     CX q2, q1;      
"""
    parser = QasmParser(qasm)

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')

    expected_circuit = Circuit()
    # CX q1[0], q1[1];
    expected_circuit.append(cirq.CNOT(q1_0, q1_1))
    # CX q1, q2[0];
    expected_circuit.append(cirq.CNOT(q1_0, q2_0))
    expected_circuit.append(cirq.CNOT(q1_1, q2_0))
    # CX q2, q1;
    expected_circuit.append(cirq.CNOT(q2_0, q1_0))
    expected_circuit.append(cirq.CNOT(q2_1, q1_1))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is False

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}


def test_CX_gate_not_enough_args():
    qasm = """
     OPENQASM 2.0;          
     qreg q[2];
     CX q[0];
"""
    parser = QasmParser(qasm)

    try:
        parser.parse()
        raise AssertionError("should fail with wrong arg length error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "CX only takes 2 args, got: 1, at line 4"


def test_cx_gate_mismatched_registers():
    qasm = """
     OPENQASM 2.0;
     qreg q1[2];
     qreg q2[3];
     CX q1, q2;
"""
    parser = QasmParser(qasm)

    try:
        parser.parse()
        raise AssertionError("should fail with mismatching registers error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "Non matching quantum registers of " \
                             "length 2 and 3 at line 5"


def test_u_gate():
    qasm = """
     OPENQASM 2.0;
     qreg q[1];
     U(pi, 2 * pi, pi / 3.0) q[0];
"""
    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(
        QasmUGate(sympy.pi / Number(3.0),
                  sympy.pi,
                  Number(2) * sympy.pi)(q0))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is False

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


@pytest.mark.parametrize(
    'expr',
    [
        '.333 + 4',
        '1.0 * 2',
        '0.1 ^ pi',
        '0.1 / pi',
        '2.0e-05 ^ (1/2)',
        '1.2E+05 * (3 + 2)',
        '123123.2132312 * cos(pi)',
        '123123.2132312 * sin(2 * pi)',
        '3 + 4 * 2',
        '3 * 4 + 2',
        '3 * 4 ^ 2',
        '3 * 4 ^ 2',
        '3 - 4 ^ 2',
        '(-1) * pi',
        '(+1) * pi',
        '-3 * 5 + 2',
        '+4 * (-3) ^ 5 - 2',
        'tan(123123.2132312)',
        'ln(pi)',
        'exp(2*pi)',
        'sqrt(4)',
        'acos(1)',
        'atan(0.2)',
        'asin(1.2)',
    ]
)
def test_expressions(expr: str):
    qasm = """
     OPENQASM 2.0;
     qreg q[1];
     U({}, 2 * pi, pi / 3.0) q[0];
""".format(expr)

    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(
        QasmUGate(sympy.pi / Number(3.0),
                  sympy.sympify(expr),
                  Number(2) * sympy.pi)(q0))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is False

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


def test_unknown_function():
    qasm = """
     OPENQASM 2.0;
     qreg q[1];
     U(nonexistent(3), 2 * pi, pi / 3.0) q[0];
"""
    parser = QasmParser(qasm)
    try:
        parser.parse()
        raise AssertionError("should fail with no format error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "Function not recognized: 'nonexistent' at line 4"


@pytest.mark.parametrize(
    'qasm_gate,cirq_gate',
    [
        ('cx', cirq.CNOT),
        ('CX', cirq.CNOT),
        ('cz', cirq.CZ),
        ('cy', cirq.ControlledGate(cirq.Y)),
        ('swap', cirq.SWAP),
    ]
)
def test_two_qubit_gates(qasm_gate: str, cirq_gate: cirq.TwoQubitGate):
    qasm = """
     OPENQASM 2.0;   
     include "qelib1.inc";       
     qreg q1[2];
     qreg q2[2];
     {0} q1[0], q1[1];
     {0} q1, q2[0];
     {0} q2, q1;      
""".format(qasm_gate)
    parser = QasmParser(qasm)

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')

    expected_circuit = Circuit()
    # CX q1[0], q1[1];
    expected_circuit.append(cirq_gate(q1_0, q1_1))
    # CX q1, q2[0];
    expected_circuit.append(cirq_gate(q1_0, q2_0))
    expected_circuit.append(cirq_gate(q1_1, q2_0))
    # CX q2, q1;
    expected_circuit.append(cirq_gate(q2_0, q1_0))
    expected_circuit.append(cirq_gate(q2_1, q1_1))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}


@pytest.mark.parametrize(
    'qasm_gate',
    [
        'cx',
        'CX',
        'cz',
        'cy',
        'swap',
    ]
)
def test_two_qubit_gates_not_enough_args(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];
     {} q[0];
""".format(qasm_gate)

    parser = QasmParser(qasm)

    try:
        parser.parse()
        raise AssertionError("should fail with wrong arg length error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "{} only takes 2 args, got: 1," \
                             " at line 5".format(qasm_gate)


def test_id_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     id q[0];
     id q;
"""

    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq.IdentityGate(1).on(q0))
    expected_circuit.append(cirq.IdentityGate(2).on(q0, q1))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


@pytest.mark.parametrize(
    'qasm_gate,cirq_gate',
    [
        ('rx', cirq.Rx),
        ('ry', cirq.Ry),
        ('rz', cirq.Rz),
    ]
)
def test_rotation_gates(qasm_gate: str, cirq_gate: cirq.SingleQubitGate):
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {} (pi/2) q[0];
     {} (pi) q;
    """.format(qasm_gate, qasm_gate)

    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq_gate(np.pi / 2).on(q0))
    expected_circuit.append(cirq.Moment([cirq_gate(np.pi).on(q0),
                                         cirq_gate(np.pi).on(q1)]))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}

## TODO: other gates
## TODO: generalize the qreg validation and assignment logic
## TODO: comments
