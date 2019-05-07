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
from sympy import Number

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

    with pytest.raises(QasmException,
                       match="Unsupported OpenQASM version: 2.1, "
                       "only 2.0 is supported currently by Cirq"):
        parser.parse()


def test_format_header_with_quelibinc_circuit():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
"""
    parser = QasmParser(qasm)

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


@pytest.mark.parametrize('qasm', [
    "include \"qelib1.inc\";",
    "",
    "qreg q[3];",
])
def test_error_not_starting_with_format(qasm: str):
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match="Missing 'OPENQASM 2.0;' statement"):
        parser.parse()


def test_comments():
    parser = QasmParser("""
    //this is the format 
    OPENQASM 2.0;
    // this is some other comment
    include "qelib1.inc";
    // and something at the end of the file
    // multiline 
    """)

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


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


@pytest.mark.parametrize('qasm', [
    """OPENQASM 2.0;
           qreg q[2];
           creg q[3];
               """,
    """OPENQASM 2.0;
           creg q[2];
           qreg q[3];
               """,
])
def test_already_defined_error(qasm: str):
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match=r"q.*already defined.* line 3"):
        parser.parse()


@pytest.mark.parametrize('qasm', [
    """OPENQASM 2.0;
           qreg q[0];
               """,
    """OPENQASM 2.0;
           creg q[0];
               """,
])
def test_zero_length_register(qasm: str):
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match="Illegal, zero-length register 'q' at line 2"):
        parser.parse()


def test_unexpected_end_of_file():
    qasm = """
                OPENQASM 2.0;
                include
           """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match="Unexpected end of file"):
        parser.parse()


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

    cirq.Simulator().run(parsed_qasm.circuit)


def test_CX_gate_not_enough_args():
    qasm = """
     OPENQASM 2.0;          
     qreg q[2];
     CX q[0];
"""
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match=r"CX only takes 2 arg\(s\) " \
                             r"\(qubits and/or registers\)" \
                             r", got: 1, at line 4"):
        parser.parse()


def test_cx_gate_mismatched_registers():
    qasm = """
     OPENQASM 2.0;
     qreg q1[2];
     qreg q2[3];
     CX q1, q2;
"""
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match=r"Non matching quantum registers of "
                       r"length \[2, 3\] at line 5"):
        parser.parse()


def test_unknown_basic_gate():
    qasm = """
         OPENQASM 2.0;          
         qreg q[2];
         foobar q[0];
    """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match=r"""Unknown gate "foobar".* line 4.*forgot.*\?"""):
        parser.parse()


def test_unknown_standard_gate():
    qasm = """
         OPENQASM 2.0;  
         include "qelib1.inc";        
         qreg q[2];
         foobar q[0];
    """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match=r"""Unknown gate "foobar" at line 5"""):
        parser.parse()


def test_syntax_error():
    qasm = """
         OPENQASM 2.0;                   
         qreg q[2] bla;
         foobar q[0];
    """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match=r"""Syntax error: 'bla'.*"""):
        parser.parse()


def test_undefined_register_from_qubit_arg():
    qasm = """
            OPENQASM 2.0;                   
            qreg q[2];
            CX q[0], q2[1];
       """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match=r"""Undefined.*register.*q2.*"""):
        parser.parse()


def test_undefined_register_from_register_arg():
    qasm = """
            OPENQASM 2.0;                   
            qreg q[2];
            qreg q2[2];
            CX q1, q2;
       """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match=r"""Undefined.*register.*q.*"""):
        parser.parse()


def test_u_gate():
    qasm = """
     OPENQASM 2.0;
     qreg q[2];
     U(pi, 2 * pi, pi / 3.0) q[0];
     U(pi, 2 * pi, pi / 3.0) q;
"""
    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(
        QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                  float(Number(2) * sympy.pi))(q0))

    expected_circuit.append(
        cirq.Moment([
            QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                      float(Number(2) * sympy.pi))(q0),
            QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                      float(Number(2) * sympy.pi))(q1)
        ]))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is False

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


def test_u3_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     u3(pi, 2 * pi, pi / 3.0) q[0];
     u3(pi, 2 * pi, pi / 3.0) q;
"""
    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(
        QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                  float(Number(2) * sympy.pi))(q0))

    expected_circuit.append(
        cirq.Moment([
            QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                      float(Number(2) * sympy.pi))(q0),
            QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.pi),
                      float(Number(2) * sympy.pi))(q1)
        ]))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


@pytest.mark.parametrize('expr', [
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
])
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
        QasmUGate(float(sympy.pi / Number(3.0)), float(sympy.sympify(expr)),
                  float(Number(2) * sympy.pi))(q0))

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

    with pytest.raises(QasmException,
                       match=r"Function not recognized:"
                       r" 'nonexistent' at line 4"):
        parser.parse()


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
    cirq.Simulator().run(parsed_qasm.circuit)


rotation_gates = [
    ('rx', cirq.Rx),
    ('ry', cirq.Ry),
    ('rz', cirq.Rz),
]


@pytest.mark.parametrize('qasm_gate,cirq_gate', rotation_gates)
def test_rotation_gates(qasm_gate: str, cirq_gate: cirq.SingleQubitGate):
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {}(pi/2) q[0];
     {}(pi) q;
    """.format(qasm_gate, qasm_gate)

    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq_gate(float(np.pi / 2)).on(q0))
    expected_circuit.append(
        cirq.Moment(
            [cirq_gate(float(np.pi)).on(q0),
             cirq_gate(float(np.pi)).on(q1)]))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_wrong_number_of_args(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];     
     {}(pi) q[0], q[1];     
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r".*{}.* takes 1 arg\(s\).*got.*2.*line 5".format(qasm_gate)):
        parser.parse()


@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_zero_params_error(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];     
     {}() q[1];     
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r".*{}.* takes 1 parameter\(s\).*got.*0.*line 5".format(
                qasm_gate)):
        parser.parse()


@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_too_much_params_error(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];     
     {}(pi, pi) q[1];     
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r".*{}.* takes 1 parameter\(s\).*got.*2.*line 5".format(
                qasm_gate)):
        parser.parse()


one_qubit_gates = [
    ('x', cirq.X),
    ('y', cirq.Y),
    ('z', cirq.Z),
    ('h', cirq.H),
    ('s', cirq.S),
    ('t', cirq.T),
    ('sdg', cirq.S**-1),
    ('tdg', cirq.T**-1),
]


@pytest.mark.parametrize('qasm_gate,cirq_gate', one_qubit_gates)
def test_single_qubit_gates(qasm_gate: str, cirq_gate: cirq.SingleQubitGate):
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {0} q[0];
     {0} q;
     {0}() q;
     {0}() q;
    """.format(qasm_gate)

    parser = QasmParser(qasm)

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq_gate.on(q0))
    expected_circuit.append(cirq.Moment([cirq_gate.on(q0), cirq_gate.on(q1)]))
    expected_circuit.append(cirq.Moment([cirq_gate.on(q0), cirq_gate.on(q1)]))
    expected_circuit.append(cirq.Moment([cirq_gate.on(q0), cirq_gate.on(q1)]))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


two_qubit_gates = [('cx', cirq.CNOT), ('CX', cirq.CNOT), ('cz', cirq.CZ),
                   ('cy', cirq.ControlledGate(cirq.Y)), ('swap', cirq.SWAP),
                   ('ch', cirq.ControlledGate(cirq.H))]


@pytest.mark.parametrize('qasm_gate,cirq_gate', two_qubit_gates)
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
    cirq.Simulator().run(parsed_qasm.circuit)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in two_qubit_gates])
def test_two_qubit_gates_not_enough_args(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];
     {} q[0];
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r".*{}.* takes 2 arg\(s\).*got.*1.*line 5".format(qasm_gate)):
        parser.parse()


@pytest.mark.parametrize('qasm_gate', [g[0] for g in two_qubit_gates])
def test_two_qubit_gates_with_too_much_parameters(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;    
     include "qelib1.inc";             
     qreg q[2];
     {}(pi) q[0];
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r".*{}.* takes 0 parameter\(s\).*got.*1.*line 5".format(
                qasm_gate)):
        parser.parse()


three_qubit_gates = [('ccx', cirq.TOFFOLI), ('cswap', cirq.CSWAP)]


@pytest.mark.parametrize('qasm_gate,cirq_gate', three_qubit_gates)
def test_three_qubit_gates(qasm_gate: str, cirq_gate: cirq.TwoQubitGate):
    qasm = """
     OPENQASM 2.0;   
     include "qelib1.inc";       
     qreg q1[2];
     qreg q2[2];
     qreg q3[2];
     {0} q1[0], q1[1], q2[0];
     {0} q1, q2[0], q3[0];
     {0} q1, q2, q3;      
""".format(qasm_gate)
    parser = QasmParser(qasm)

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')
    q3_0 = cirq.NamedQubit('q3_0')
    q3_1 = cirq.NamedQubit('q3_1')

    expected_circuit = Circuit()

    expected_circuit.append(cirq_gate(q1_0, q1_1, q2_0))

    expected_circuit.append(cirq_gate(q1_0, q2_0, q3_0))
    expected_circuit.append(cirq_gate(q1_1, q2_0, q3_0))

    expected_circuit.append(cirq_gate(q1_0, q2_0, q3_0))
    expected_circuit.append(cirq_gate(q1_1, q2_1, q3_1))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2, 'q3': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in three_qubit_gates])
def test_three_qubit_gates_not_enough_args(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {} q[0];
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r""".*{}.* takes 3 arg\(s\).*got.*1.*line 5""".format(
                qasm_gate)):
        parser.parse()


@pytest.mark.parametrize('qasm_gate', [g[0] for g in three_qubit_gates])
def test_three_qubit_gates_with_too_much_parameters(qasm_gate: str):
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {}(pi) q[0];
""".format(qasm_gate)

    parser = QasmParser(qasm)

    with pytest.raises(
            QasmException,
            match=r""".*{}.*parameter.*line 5.*""".format(qasm_gate)):
        parser.parse()


def test_measure_individual_bits():
    qasm = """
         OPENQASM 2.0;   
         include "qelib1.inc";       
         qreg q1[2];
         creg c1[2];                        
         measure q1[0] -> c1[0];
         measure q1[1] -> c1[1];
    """
    parser = QasmParser(qasm)

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')

    expected_circuit = Circuit()

    expected_circuit.append(
        cirq.MeasurementGate(num_qubits=1, key='c1_1').on(q1_1))
    expected_circuit.append(
        cirq.MeasurementGate(num_qubits=1, key='c1_0').on(q1_0))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2}
    assert parsed_qasm.cregs == {'c1': 2}
    cirq.Simulator().run(parsed_qasm.circuit)


def test_measure_registers():
    qasm = """
         OPENQASM 2.0;   
         include "qelib1.inc";       
         qreg q1[3];
         creg c1[3];                        
         measure q1 -> c1;       
    """
    parser = QasmParser(qasm)

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q1_2 = cirq.NamedQubit('q1_2')

    expected_circuit = Circuit()

    expected_circuit.append(
        cirq.MeasurementGate(num_qubits=1, key='c1_0').on(q1_0))
    expected_circuit.append(
        cirq.MeasurementGate(num_qubits=1, key='c1_1').on(q1_1))
    expected_circuit.append(
        cirq.MeasurementGate(num_qubits=1, key='c1_2').on(q1_2))

    parsed_qasm = parser.parse()

    assert parsed_qasm.supportedFormat is True
    assert parsed_qasm.qelib1Include is True

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 3}
    assert parsed_qasm.cregs == {'c1': 3}
    cirq.Simulator().run(parsed_qasm.circuit)


def test_measure_mismatched_register_size():
    qasm = """
         OPENQASM 2.0;   
         include "qelib1.inc";       
         qreg q1[2];
         creg c1[3];                        
         measure q1 -> c1;       
    """

    parser = QasmParser(qasm)

    with pytest.raises(QasmException,
                       match=r""".*mismatched register sizes 2 -> 3.*line 6"""):
        parser.parse()
