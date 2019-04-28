#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import pytest

from cirq import Circuit
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
import cirq.testing as ct


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
        pass


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
    "",
    "include \"qelib1.inc\";"
])
def test_error_not_starting_with_format(qasm: str):
    parser = QasmParser(qasm)
    try:
        parser.parse()
        raise AssertionError("should fail with no format error")
    except QasmException as ex:
        assert ex.qasm == qasm
        assert ex.message == "Missing 'OPENQASM 2.0;' statement"
        pass


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
