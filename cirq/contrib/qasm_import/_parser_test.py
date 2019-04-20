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
                include "qelib1.inc";
                creg
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


def test_syntax_error():
    qasm = """
         OPENQASM 2.0;                   
         qreg q[2] bla;
         foobar q[0];
    """
    parser = QasmParser(qasm)

    with pytest.raises(QasmException, match=r"""Syntax error: 'bla'.*"""):
        parser.parse()
