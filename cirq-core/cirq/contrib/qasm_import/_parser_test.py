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

import re
import textwrap
from typing import Callable

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
from cirq.testing import consistent_qasm as cq


def test_format_header_circuit():
    parser = QasmParser()

    parsed_qasm = parser.parse("OPENQASM 2.0;")

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


def test_unsupported_format():
    qasm = "OPENQASM 2.1;"
    parser = QasmParser()

    with pytest.raises(QasmException, match="Unsupported.*2.1.*2.0.*supported.*"):
        parser.parse(qasm)


def test_format_header_with_quelibinc_circuit():
    qasm = """OPENQASM 2.0;
include "qelib1.inc";
"""
    parser = QasmParser()

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


@pytest.mark.parametrize('qasm', ["include \"qelib1.inc\";", "", "qreg q[3];"])
def test_error_not_starting_with_format(qasm: str):
    parser = QasmParser()

    with pytest.raises(QasmException, match="Missing 'OPENQASM 2.0;' statement"):
        parser.parse(qasm)


def test_comments():
    parser = QasmParser()

    parsed_qasm = parser.parse(
        """
    //this is the format
    OPENQASM 2.0;
    // this is some other comment
    include "qelib1.inc";
    // and something at the end of the file
    // multiline
    """
    )

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())


def test_multiple_qreg_declaration():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg a_quantum_register [ 1337 ];
     qreg q[42];
"""
    parser = QasmParser()

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())
    assert parsed_qasm.qregs == {'a_quantum_register': 1337, 'q': 42}


@pytest.mark.parametrize(
    'qasm',
    [
        """OPENQASM 2.0;
           qreg q[2];
           creg q[3];
               """,
        """OPENQASM 2.0;
           creg q[2];
           qreg q[3];
               """,
    ],
)
def test_already_defined_error(qasm: str):
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"q.*already defined.* line 3"):
        parser.parse(qasm)


@pytest.mark.parametrize(
    'qasm',
    [
        """OPENQASM 2.0;
           qreg q[0];
               """,
        """OPENQASM 2.0;
           creg q[0];
               """,
    ],
)
def test_zero_length_register(qasm: str):
    parser = QasmParser()

    with pytest.raises(QasmException, match=".* zero-length.*'q'.*line 2"):
        parser.parse(qasm)


def test_unexpected_end_of_file():
    qasm = """OPENQASM 2.0;
              include "qelib1.inc";
              creg
           """
    parser = QasmParser()

    with pytest.raises(QasmException, match="Unexpected end of file"):
        parser.parse(qasm)


def test_multiple_creg_declaration():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     creg a_classical_register [1337];
     qreg a_quantum_register [1337];
     creg c[42];
"""
    parser = QasmParser()

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, Circuit())
    assert parsed_qasm.qregs == {'a_quantum_register': 1337}
    assert parsed_qasm.cregs == {'a_classical_register': 1337, 'c': 42}


def test_syntax_error():
    qasm = """OPENQASM 2.0;
         qreg q[2] bla;
         foobar q[0];
    """
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Syntax error: 'bla'.*"""):
        parser.parse(qasm)


def test_CX_gate():
    qasm = """OPENQASM 2.0;
     qreg q1[2];
     qreg q2[2];
     CX q1[0], q1[1];
     CX q1, q2[0];
     CX q2, q1;
"""
    parser = QasmParser()

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

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}


def test_classical_control():
    qasm = """OPENQASM 2.0;
        qreg q[2];
        creg a[1];
        measure q[0] -> a[0];
        if (a==1) CX q[0],q[1];
    """
    parser = QasmParser()

    q_0 = cirq.NamedQubit('q_0')
    q_1 = cirq.NamedQubit('q_1')
    expected_circuit = cirq.Circuit(
        cirq.measure(q_0, key='a_0'),
        cirq.CNOT(q_0, q_1).with_classical_controls(sympy.Eq(sympy.Symbol('a_0'), 1)),
    )

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}

    # Note this cannot *exactly* round-trip because the way QASM and Cirq handle measurements
    # into classical registers is different. Cirq parses QASM classical registers into m_a_i for i
    # in 0..bit_count. Thus the generated key has an extra "_0" at the end.
    expected_generated_qasm = f"""// Generated from Cirq v{cirq.__version__}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q_0, q_1]
qreg q[2];
creg m_a_0[1];


measure q[0] -> m_a_0[0];
if (m_a_0==1) cx q[0],q[1];
"""
    assert cirq.qasm(parsed_qasm.circuit) == expected_generated_qasm


def test_classical_control_multi_bit():
    qasm = """OPENQASM 2.0;
        qreg q[2];
        creg a[2];
        measure q[0] -> a[0];
        measure q[0] -> a[1];
        if (a==1) CX q[0],q[1];
    """
    parser = QasmParser()

    q_0 = cirq.NamedQubit('q_0')
    q_1 = cirq.NamedQubit('q_1')

    # Since we split the measurement into two, we also need two conditions.
    # m_a==1 corresponds to m_a[0]==1, m_a[1]==0
    expected_circuit = cirq.Circuit(
        cirq.measure(q_0, key='a_0'),
        cirq.measure(q_0, key='a_1'),
        cirq.CNOT(q_0, q_1).with_classical_controls(
            sympy.Eq(sympy.Symbol('a_0'), 1), sympy.Eq(sympy.Symbol('a_1'), 0)
        ),
    )

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}

    # Note that this will *not* round-trip, but there's no good way around that due to the
    # difference in how Cirq and QASM do multi-bit measurements.
    with pytest.raises(ValueError, match='QASM does not support multiple conditions'):
        _ = cirq.qasm(parsed_qasm.circuit)


def test_CX_gate_not_enough_args():
    qasm = """OPENQASM 2.0;
     qreg q[2];
     CX q[0];
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"CX.*takes.*got.*1.*line 3"):
        parser.parse(qasm)


def test_CX_gate_mismatched_registers():
    qasm = """OPENQASM 2.0;
     qreg q1[2];
     qreg q2[3];
     CX q1, q2;
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"Non matching.*length \[2 3\].*line 4"):
        parser.parse(qasm)


def test_CX_gate_bounds():
    qasm = """OPENQASM 2.0;
     qreg q1[2];
     qreg q2[3];
     CX q1[4], q2[0];
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"Out of bounds.*4.*q1.*2.*line 4"):
        parser.parse(qasm)


def test_CX_gate_arg_overlap():
    qasm = """OPENQASM 2.0;
     qreg q1[2];
     qreg q2[3];
     CX q1[1], q1[1];
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"Overlapping.*at line 4"):
        parser.parse(qasm)


def test_U_gate():
    qasm = """
     OPENQASM 2.0;
     qreg q[2];
     U(pi, 2.3, 3) q[0];
     U(+3.14, -pi, (8)) q;
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(
        cirq.Moment(
            [
                QasmUGate(1.0, 2.3 / np.pi, 3 / np.pi)(q0),
                QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q1),
            ]
        )
    )

    expected_circuit.append(cirq.Moment([QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q0)]))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


def test_U_angles():
    qasm = """
    OPENQASM 2.0;
    qreg q[1];
    U(pi/2,0,pi) q[0];
    """

    c = QasmParser().parse(qasm).circuit
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(c), cirq.unitary(cirq.H), atol=1e-7
    )


def test_U_gate_zero_params_error():
    qasm = """OPENQASM 2.0;
     qreg q[2];
     U q[1];"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=r"U takes 3.*got.*0.*line 3"):
        parser.parse(qasm)


def test_U_gate_too_much_params_error():
    qasm = """OPENQASM 2.0;
     qreg q[2];
     U(pi, pi, pi, pi) q[1];"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=r"U takes 3.*got.*4.*line 3"):
        parser.parse(qasm)


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
        '3 - 4 * 2',  # precedence of *
        '3 * 4 + 2',  # precedence of *
        '3 * 4 ^ 2',  # precedence of ^
        '3 - 4 ^ 2',  # precedence of ^
        '3^2^(-2)',  # right associativity of ^
        '(-1) * pi',
        '(+1) * pi',
        '-3 * 5 + 2',
        '(+4 * (-3) ^ 5 - 2)',
        'tan(123123.2132312)',
        'ln(pi)',
        'exp(2*pi)',
        'sqrt(4)',
        'acos(1)',
        'atan(0.2)',
    ],
)
def test_expressions(expr: str):
    qasm = f"""OPENQASM 2.0;
     qreg q[1];
     U({expr}, 2 * pi, pi / 2.0) q[0];
"""

    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(QasmUGate(float(sympy.sympify(expr)) / np.pi, 2.0, 1 / 2.0)(q0))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert not parsed_qasm.qelib1Include

    ct.assert_allclose_up_to_global_phase(
        cirq.unitary(parsed_qasm.circuit), cirq.unitary(expected_circuit), atol=1e-10
    )
    assert parsed_qasm.qregs == {'q': 1}


def test_unknown_function():
    qasm = """OPENQASM 2.0;
     qreg q[1];
     U(nonexistent(3), 2 * pi, pi / 3.0) q[0];
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r".*not recognized.*'nonexistent'.*line 3"):
        parser.parse(qasm)


rotation_gates = [('rx', cirq.rx), ('ry', cirq.ry), ('rz', cirq.rz)]


single_qubit_gates = [
    ('x', cirq.X),
    ('y', cirq.Y),
    ('z', cirq.Z),
    ('h', cirq.H),
    ('s', cirq.S),
    ('t', cirq.T),
    ('sdg', cirq.S**-1),
    ('tdg', cirq.T**-1),
    ('sx', cirq.XPowGate(exponent=0.5)),
]


@pytest.mark.parametrize('qasm_gate,cirq_gate', rotation_gates)
def test_rotation_gates(qasm_gate: str, cirq_gate: Callable[[float], cirq.Gate]):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate}(pi/2) q[0];
     {qasm_gate}(pi) q;
    """

    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq.Moment([cirq_gate(np.pi / 2).on(q0), cirq_gate(np.pi).on(q1)]))
    expected_circuit.append(cirq.Moment([cirq_gate(np.pi).on(q0)]))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_wrong_number_of_args(qasm_gate: str):
    qasm = f"""
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate}(pi) q[0], q[1];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=f".*{qasm_gate}.* takes 1.*got.*2.*line 5"):
        parser.parse(qasm)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in rotation_gates])
def test_rotation_gates_zero_params_error(qasm_gate: str):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[1];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=f".*{qasm_gate}.* takes 1.*got.*0.*line 4"):
        parser.parse(qasm)


def test_qelib_gate_without_include_statement():
    qasm = """OPENQASM 2.0;
         qreg q[2];
         x q[0];
    """
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Unknown gate "x".* line 3.*forget.*\?"""):
        parser.parse(qasm)


def test_undefined_register_from_qubit_arg():
    qasm = """OPENQASM 2.0;
            qreg q[2];
            CX q[0], q2[1];
       """
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Undefined.*register.*q2.*"""):
        parser.parse(qasm)


def test_undefined_register_from_register_arg():
    qasm = """OPENQASM 2.0;
            qreg q[2];
            qreg q2[2];
            CX q1, q2;
       """
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Undefined.*register.*q.*"""):
        parser.parse(qasm)


def test_measure_individual_bits():
    qasm = """
         OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[2];
         creg c1[2];
         measure q1[0] -> c1[0];
         measure q1[1] -> c1[1];
    """
    parser = QasmParser()

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')

    expected_circuit = Circuit()

    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_0').on(q1_0))
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_1').on(q1_1))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2}
    assert parsed_qasm.cregs == {'c1': 2}


def test_measure_registers():
    qasm = """OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[3];
         creg c1[3];
         measure q1 -> c1;
    """
    parser = QasmParser()

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q1_2 = cirq.NamedQubit('q1_2')

    expected_circuit = Circuit()

    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_0').on(q1_0))
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_1').on(q1_1))
    expected_circuit.append(cirq.MeasurementGate(num_qubits=1, key='c1_2').on(q1_2))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 3}
    assert parsed_qasm.cregs == {'c1': 3}


def test_measure_mismatched_register_size():
    qasm = """OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[2];
         creg c1[3];
         measure q1 -> c1;
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=r""".*mismatched .* 2 -> 3.*line 5"""):
        parser.parse(qasm)


def test_measure_to_quantum_register():
    qasm = """OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[3];
         qreg q2[3];
         creg c1[3];
         measure q2 -> q1;
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Undefined classical register.*q1.*line 6"""):
        parser.parse(qasm)


def test_measure_undefined_classical_bit():
    qasm = """OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[3];
         creg c1[3];
         measure q1[1] -> c2[1];
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Undefined classical register.*c2.*line 5"""):
        parser.parse(qasm)


def test_measure_from_classical_register():
    qasm = """OPENQASM 2.0;
         include "qelib1.inc";
         qreg q1[2];
         creg c1[3];
         creg c2[3];
         measure c1 -> c2;
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=r"""Undefined quantum register.*c1.*line 6"""):
        parser.parse(qasm)


def test_measurement_bounds():
    qasm = """OPENQASM 2.0;
     qreg q1[3];
     creg c1[3];
     measure q1[0] -> c1[4];
"""
    parser = QasmParser()

    with pytest.raises(QasmException, match=r"Out of bounds bit.*4.*c1.*size 3.*line 4"):
        parser.parse(qasm)


def test_reset():
    qasm = textwrap.dedent(
        """\
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        reset q[0];
        measure q[0] -> c[0];
        """
    )

    parser = QasmParser()

    q_0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit([cirq.X(q_0), cirq.reset(q_0), cirq.measure(q_0, key='c_0')])

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}
    assert parsed_qasm.cregs == {'c': 1}


def test_u1_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     u1(pi / 3.0) q[0];
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(QasmUGate(0, 0, 1.0 / 3.0)(q0))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


def test_u2_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     u2(2 * pi, pi / 3.0) q[0];
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(QasmUGate(0.5, 2.0, 1.0 / 3.0)(q0))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


def test_id_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     id q;
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(cirq.IdentityGate(num_qubits=1)(q0))
    expected_circuit.append(cirq.IdentityGate(num_qubits=1)(q1))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


def test_u3_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     u3(pi, 2.3, 3) q[0];
     u3(+3.14, -pi, (8)) q;
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit()
    expected_circuit.append(
        cirq.Moment(
            [
                QasmUGate(1.0, 2.3 / np.pi, 3 / np.pi)(q0),
                QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q1),
            ]
        )
    )

    expected_circuit.append(cirq.Moment([QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q0)]))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


def test_r_gate():
    qasm = """
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     r(pi, pi / 2.0) q[0];
"""
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit()
    expected_circuit.append(QasmUGate(1.0, 0.0, 0.0)(q0))

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


@pytest.mark.parametrize(
    'qasm_gate',
    ['id', 'u2', 'u3', 'r'] + [g[0] for g in rotation_gates] + [g[0] for g in single_qubit_gates],
)
def test_standard_single_qubit_gates_wrong_number_of_args(qasm_gate):
    qasm = f"""
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[0], q[1];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=r".* takes 1.*got.*2.*line 5"):
        parser.parse(qasm)


@pytest.mark.parametrize(
    ['qasm_gate', 'num_params'],
    [['id', 0], ['u2', 2], ['u3', 3], ['rx', 1], ['ry', 1], ['rz', 1], ['r', 2]]
    + [[g[0], 0] for g in single_qubit_gates],
)
def test_standard_gates_wrong_params_error(qasm_gate: str, num_params: int):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate}(pi, 2*pi, 3*pi, 4*pi, 5*pi) q[1];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=f".*{qasm_gate}.* takes {num_params}.*got.*5.*line 4"):
        parser.parse(qasm)

    if num_params == 0:
        return

    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[1];
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=f".*{qasm_gate}.* takes {num_params}.*got.*0.*line 4"):
        parser.parse(qasm)


two_qubit_gates = [
    ('cx', cirq.CNOT),
    ('CX', cirq.CNOT),
    ('cz', cirq.CZ),
    ('cy', cirq.ControlledGate(cirq.Y)),
    ('swap', cirq.SWAP),
    ('ch', cirq.ControlledGate(cirq.H)),
]


# Mapping of two-qubit gates and `num_params`
two_qubit_param_gates = {
    ('cu1', cirq.ControlledGate(QasmUGate(0, 0, 0.1 / np.pi))): 1,
    ('cu3', cirq.ControlledGate(QasmUGate(0.1 / np.pi, 0.2 / np.pi, 0.3 / np.pi))): 3,
    ('crz', cirq.ControlledGate(cirq.rz(0.1))): 1,
}


@pytest.mark.parametrize('qasm_gate,cirq_gate', two_qubit_gates)
def test_two_qubit_gates(qasm_gate: str, cirq_gate: cirq.testing.TwoQubitGate):
    qasm = f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q1[2];
    qreg q2[2];
    {qasm_gate} q1[0], q1[1];
    {qasm_gate} q1, q2[0];
    {qasm_gate} q2, q1;
"""
    parser = QasmParser()

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

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}


@pytest.mark.parametrize(
    'qasm_gate,cirq_gate,num_params',
    [
        (gate_map[0], gate_map[1], num_param)
        for gate_map, num_param in two_qubit_param_gates.items()
    ],
)
def test_two_qubit_param_gates(
    qasm_gate: str, cirq_gate: cirq.testing.TwoQubitGate, num_params: int
):
    params = '(0.1, 0.2, 0.3)' if num_params == 3 else '(0.1)'
    qasm = f"""
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q1[2];
    qreg q2[2];
    {qasm_gate}{params} q1[0], q1[1];
    {qasm_gate}{params} q1, q2[0];
    {qasm_gate}{params} q2, q1;
    """
    parser = QasmParser()

    q1_0 = cirq.NamedQubit('q1_0')
    q1_1 = cirq.NamedQubit('q1_1')
    q2_0 = cirq.NamedQubit('q2_0')
    q2_1 = cirq.NamedQubit('q2_1')

    expected_circuit = cirq.Circuit()
    expected_circuit.append(cirq_gate.on(q1_0, q1_1))
    expected_circuit.append(cirq_gate.on(q1_0, q2_0))
    expected_circuit.append(cirq_gate.on(q1_1, q2_0))
    expected_circuit.append(cirq_gate.on(q2_0, q1_0))
    expected_circuit.append(cirq_gate.on(q2_1, q1_1))
    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2}


@pytest.mark.parametrize(
    'qasm_gate', [g[0] for g in two_qubit_gates] + [g[0] for g in two_qubit_param_gates.keys()]
)
def test_two_qubit_gates_not_enough_qubits(qasm_gate: str):
    if qasm_gate in ('cu1', 'crz'):
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        {qasm_gate}(0.1) q[0];
    """
    elif qasm_gate == 'cu3':
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        {qasm_gate}(0.1, 0.2, 0.3) q[0];
    """
    else:
        qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        {qasm_gate} q[0];
    """

    parser = QasmParser()

    with pytest.raises(QasmException, match=rf".*{qasm_gate}.* takes 2 arg\(s\).*got.*1.*line 5"):
        parser.parse(qasm)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in two_qubit_param_gates.keys()])
def test_two_qubit_gates_not_enough_args(qasm_gate: str):
    qasm = f"""
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[0];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=rf".*{qasm_gate}.* takes 2 arg\(s\).*got.*1.*line 5"):
        parser.parse(qasm)


@pytest.mark.parametrize(
    'qasm_gate', [g[0] for g in two_qubit_gates] + [g[0] for g in two_qubit_param_gates.keys()]
)
def test_two_qubit_gates_with_too_much_parameters(qasm_gate: str):
    if qasm_gate in ('cu1', 'cu3', 'crz'):
        num_params_needed = 3 if qasm_gate == 'cu3' else 1
    else:
        num_params_needed = 0

    qasm = f"""
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        {qasm_gate}(pi, pi/2, pi/3, pi/4) q[0],q[1];
    """

    parser = QasmParser()

    with pytest.raises(
        QasmException,
        match=rf".*{qasm_gate}*. takes {num_params_needed} parameter\(s\).*got.*4.*line 5",
    ):
        parser.parse(qasm)


three_qubit_gates = [('ccx', cirq.TOFFOLI), ('cswap', cirq.CSWAP)]


@pytest.mark.parametrize('qasm_gate,cirq_gate', three_qubit_gates)
def test_three_qubit_gates(qasm_gate: str, cirq_gate: cirq.testing.TwoQubitGate):
    qasm = f"""
     OPENQASM 2.0;
     include "qelib1.inc";
     qreg q1[2];
     qreg q2[2];
     qreg q3[2];
     {qasm_gate} q1[0], q1[1], q2[0];
     {qasm_gate} q1, q2[0], q3[0];
     {qasm_gate} q1, q2, q3;
"""
    parser = QasmParser()

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

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q1': 2, 'q2': 2, 'q3': 2}


@pytest.mark.parametrize('qasm_gate', [g[0] for g in three_qubit_gates])
def test_three_qubit_gates_not_enough_args(qasm_gate: str):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[0];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=rf".*{qasm_gate}.* takes 3 arg\(s\).*got.*1.*line 4"):
        parser.parse(qasm)


@pytest.mark.parametrize('qasm_gate', [g[0] for g in three_qubit_gates])
def test_three_qubit_gates_with_too_much_parameters(qasm_gate: str):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[3];
     {qasm_gate}(pi) q[0],q[1],q[2];
"""

    parser = QasmParser()

    with pytest.raises(QasmException, match=f".*{qasm_gate}.*parameter.*line 4.*"):
        parser.parse(qasm)


@pytest.mark.parametrize('qasm_gate,cirq_gate', single_qubit_gates)
def test_single_qubit_gates(qasm_gate: str, cirq_gate: cirq.Gate):
    qasm = f"""OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     {qasm_gate} q[0];
     {qasm_gate} q;
    """

    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')

    expected_circuit = Circuit([cirq_gate.on(q0), cirq_gate.on(q0), cirq_gate.on(q1)])

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


def test_openqasm_3_0_qubits():
    qasm = """OPENQASM 3.0;
     include "stdgates.inc";
     qubit[2] q;
     bit[2] b;

     x q[0];

     b[0] = measure q[0];
     reset q[0];
    """
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit([cirq.X.on(q0), cirq.measure(q0, key='b_0'), cirq.reset(q0)])

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}


def test_openqasm_3_0_scalar_qubit():
    qasm = """OPENQASM 3.0;
     include "stdgates.inc";
     qubit q;
     bit b;

     x q;

     b = measure q;
     reset q;
    """
    parser = QasmParser()

    q0 = cirq.NamedQubit('q_0')

    expected_circuit = Circuit([cirq.X.on(q0), cirq.measure(q0, key='b_0'), cirq.reset(q0)])

    parsed_qasm = parser.parse(qasm)

    assert parsed_qasm.supportedFormat

    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 1}


def test_custom_gate():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     gate g q0, q1 {
        x q0;
        y q0;
        z q1;
     }
     g q[0], q[1];
     g q[1], q[0];
    """

    # The gate definition should translate to this
    q0, q1 = cirq.NamedQubit.range(2, prefix='q')
    g = cirq.FrozenCircuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q1))

    # The outer circuit should then translate to this
    q_0, q_1 = cirq.NamedQubit.range(2, prefix='q_')  # The outer qreg array
    expected = cirq.Circuit(
        cirq.CircuitOperation(g, qubit_map={q0: q_0, q1: q_1}),
        cirq.CircuitOperation(g, qubit_map={q0: q_1, q1: q_0}),
    )

    # Verify
    parser = QasmParser()
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.circuit == expected

    # Sanity check that this unrolls to a valid circuit
    unrolled_expected = cirq.Circuit(
        cirq.X(q_0), cirq.Y(q_0), cirq.Z(q_1), cirq.X(q_1), cirq.Y(q_1), cirq.Z(q_0)
    )
    unrolled = cirq.align_left(cirq.unroll_circuit_op(parsed_qasm.circuit, tags_to_check=None))
    assert unrolled == unrolled_expected

    # Sanity check that these have the same unitaries as the QASM.
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(parsed_qasm.circuit))
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(unrolled))


def test_custom_gate_parameterized():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     gate g(p0, p1) q0, q1 {
        rx(p0) q0;
        ry(p0+p1+3) q0;
        rz(p1) q1;
     }
     g(1,2) q[0], q[1];
     g(0,4) q[1], q[0];
    """

    # The gate definition should translate to this
    p0, p1 = sympy.symbols('p0, p1')
    q0, q1 = cirq.NamedQubit.range(2, prefix='q')
    g = cirq.FrozenCircuit(
        cirq.Rx(rads=p0).on(q0), cirq.Ry(rads=p0 + p1 + 3).on(q0), cirq.Rz(rads=p1).on(q1)
    )

    # The outer circuit should then translate to this
    q_0, q_1 = cirq.NamedQubit.range(2, prefix='q_')  # The outer qreg array
    expected = cirq.Circuit(
        cirq.CircuitOperation(g, qubit_map={q0: q_0, q1: q_1}, param_resolver={'p0': 1, 'p1': 2}),
        cirq.CircuitOperation(g, qubit_map={q0: q_1, q1: q_0}, param_resolver={'p0': 0, 'p1': 4}),
    )

    # Verify
    parser = QasmParser()
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.circuit == expected

    # Sanity check that this unrolls to a valid circuit
    unrolled_expected = cirq.Circuit(
        cirq.Rx(rads=1).on(q_0),
        cirq.Ry(rads=6).on(q_0),
        cirq.Rz(rads=2).on(q_1),
        cirq.Rx(rads=0).on(q_1),
        cirq.Ry(rads=7).on(q_1),
        cirq.Rz(rads=4).on(q_0),
    )
    unrolled = cirq.align_left(cirq.unroll_circuit_op(parsed_qasm.circuit, tags_to_check=None))
    assert unrolled == unrolled_expected

    # Sanity check that these have the same unitaries as the QASM.
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(parsed_qasm.circuit))
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(unrolled))


def test_custom_gate_broadcast():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[3];
     gate g q0 {
        x q0;
        y q0;
        z q0;
     }
     g q; // broadcast to all qubits in register
    """

    # The gate definition should translate to this
    q0 = cirq.NamedQubit('q0')
    g = cirq.FrozenCircuit(cirq.X(q0), cirq.Y(q0), cirq.Z(q0))

    # The outer circuit should then translate to this
    q_0, q_1, q_2 = cirq.NamedQubit.range(3, prefix='q_')  # The outer qreg array
    expected = cirq.Circuit(
        # It is broadcast to all qubits in the qreg
        cirq.CircuitOperation(g, qubit_map={q0: q_0}),
        cirq.CircuitOperation(g, qubit_map={q0: q_1}),
        cirq.CircuitOperation(g, qubit_map={q0: q_2}),
    )

    # Verify
    parser = QasmParser()
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.circuit == expected

    # Sanity check that this unrolls to a valid circuit
    unrolled_expected = cirq.Circuit(
        cirq.X(q_0),
        cirq.Y(q_0),
        cirq.Z(q_0),
        cirq.X(q_1),
        cirq.Y(q_1),
        cirq.Z(q_1),
        cirq.X(q_2),
        cirq.Y(q_2),
        cirq.Z(q_2),
    )
    unrolled = cirq.align_left(cirq.unroll_circuit_op(parsed_qasm.circuit, tags_to_check=None))
    assert unrolled == unrolled_expected

    # Sanity check that these have the same unitaries as the QASM.
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(parsed_qasm.circuit))
    cq.assert_qiskit_parsed_qasm_consistent_with_unitary(qasm, cirq.unitary(unrolled))


def test_custom_gate_undefined_qubit_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g q0 { x q1; }
     g q
    """
    _test_parse_exception(
        qasm,
        cirq_err="Undefined qubit 'q1' at line 4",
        qiskit_err="4,19: 'q1' is not defined in this scope",
    )


def test_custom_gate_qubit_scope_closure_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g q0 { x q; }
     g q
    """
    _test_parse_exception(
        qasm,
        cirq_err="'q' is a register, not a qubit at line 4",
        qiskit_err="4,19: 'q' is a quantum register, not a qubit",
    )


def test_custom_gate_qubit_index_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g q0 { x q0[0]; }
     g q
    """
    _test_parse_exception(
        qasm,
        cirq_err="Unsupported indexed qreg 'q0[0]' at line 4",
        qiskit_err="4,21: needed ';', but instead saw [",
    )


def test_custom_gate_qreg_count_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[2];
     gate g q0 { x q0; }
     g q[0], q[1];
    """
    _test_parse_exception(
        qasm,
        cirq_err="Wrong number of qregs for 'g' at line 5",
        qiskit_err="5,5: 'g' takes 1 quantum argument, but got 2",
    )


def test_custom_gate_missing_param_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g(p) q0 { rx(p) q0; }
     g q;
    """
    _test_parse_exception(
        qasm,
        cirq_err="Wrong number of params for 'g' at line 5",
        qiskit_err=None,  # Qiskit bug? It's an invalid circuit that won't simulate.
    )


def test_custom_gate_extra_param_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g q0 { x q0; }
     g(3) q;
    """
    _test_parse_exception(
        qasm,
        cirq_err="Wrong number of params for 'g' at line 5",
        qiskit_err="5,5: 'g' takes 0 parameters, but got 1",
    )


def test_custom_gate_undefined_param_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate g q0 { rx(p) q0; }
     g q;
    """
    _test_parse_exception(
        qasm,
        cirq_err="Undefined parameter 'p' in line 4",
        qiskit_err="4,20: 'p' is not a parameter",
    )


def test_top_level_param_error():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     rx(p) q;
    """
    _test_parse_exception(
        qasm,
        cirq_err="Parameter 'p' in line 4 not supported",
        qiskit_err="4,8: 'p' is not a parameter",
    )


def _test_parse_exception(qasm: str, cirq_err: str, qiskit_err: str | None):
    parser = QasmParser()
    with pytest.raises(QasmException, match=re.escape(cirq_err)):
        parser.parse(qasm)
    pytest.importorskip("qiskit")
    import qiskit.qasm2

    if qiskit_err is None:
        qiskit.QuantumCircuit.from_qasm_str(qasm)
        return
    with pytest.raises(qiskit.qasm2.exceptions.QASM2ParseError, match=re.escape(qiskit_err)):
        qiskit.QuantumCircuit.from_qasm_str(qasm)


def test_nested_custom_gate_has_keyword_in_name():
    qasm = """OPENQASM 2.0;
     include "qelib1.inc";
     qreg q[1];
     gate gateGate qb { x qb; }
     gate qregGate qa { gateGate qa; }
     qregGate q;
    """
    qb = cirq.NamedQubit('qb')
    inner = cirq.FrozenCircuit(cirq.X(qb))
    qa = cirq.NamedQubit('qa')
    middle = cirq.FrozenCircuit(cirq.CircuitOperation(inner, qubit_map={qb: qa}))
    q_0 = cirq.NamedQubit('q_0')
    expected = cirq.Circuit(cirq.CircuitOperation(middle, qubit_map={qa: q_0}))
    parser = QasmParser()
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.circuit == expected
