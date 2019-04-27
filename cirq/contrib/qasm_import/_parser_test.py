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


def test_format_header_with_quelibinc_circuit():
    parsed_qasm = """OPENQASM 2.0;
include "qelib1.inc";
"""
    parser = QasmParser(parsed_qasm)

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
