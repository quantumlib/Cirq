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

from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._lexer import QasmLexer


def test_empty_circuit():
    assert QasmLexer("").token() is None


@pytest.mark.parametrize('number', ["00000", "03", "3", "0045", "21"])
def test_natural_numbers(number: str):
    token = QasmLexer(number).token()
    assert token is not None
    assert token.type == "NATURAL_NUMBER"
    assert token.value == int(number)


def test_supported_format():
    token = QasmLexer("OPENQASM 2.0;").token()
    assert token is not None
    assert token.type == "FORMAT_SPEC"
    assert token.value == '2.0'


def test_qelib_inc():
    token = QasmLexer('include "qelib1.inc";').token()
    assert token is not None
    assert token.type == "QELIBINC"
    assert token.value == 'include "qelib1.inc";'


def test_measurment():
    lexer = QasmLexer("measure q -> c;")
    token = lexer.token()
    assert token.type == "MEASURE"
    assert token.value == 'measure'

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == 'q'

    token = lexer.token()
    assert token.type == "ARROW"
    assert token.value == '->'

    token = lexer.token()
    assert token.type == "ID"
    assert token.value == 'c'

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ';'


@pytest.mark.parametrize(
    'identifier',
    ['b', 'CX', 'abc', 'aXY03', 'a_valid_name_with_02_digits_and_underscores'])
def test_valid_ids(identifier: str):
    token = QasmLexer(identifier).token()

    assert token is not None
    assert token.type == "ID"
    assert token.value == identifier


@pytest.mark.parametrize(
    'number', ['.333', '1.0', '0.1', '2.0e-05', '1.2E+05', '123123.2132312'])
def test_reals(number: str):
    token = QasmLexer(number).token()

    assert token is not None
    assert token.type == "NUMBER"
    assert token.value == Number(number)


def test_pi():
    lexer = QasmLexer('pi')
    token = lexer.token()
    assert token.type == "PI"
    assert token.value == sympy.pi


def test_qreg():
    lexer = QasmLexer('qreg [5];')
    token = lexer.token()
    assert token.type == "QREG"
    assert token.value == "qreg"

    token = lexer.token()
    assert token.type == "["
    assert token.value == "["

    token = lexer.token()
    assert token.type == "NATURAL_NUMBER"
    assert token.value == 5

    token = lexer.token()
    assert token.type == "]"
    assert token.value == "]"

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ";"


def test_creg():
    lexer = QasmLexer('creg [8];')
    token = lexer.token()
    assert token.type == "CREG"
    assert token.value == "creg"

    token = lexer.token()
    assert token.type == "["
    assert token.value == "["

    token = lexer.token()
    assert token.type == "NATURAL_NUMBER"
    assert token.value == 8

    token = lexer.token()
    assert token.type == "]"
    assert token.value == "]"

    token = lexer.token()
    assert token.type == ";"
    assert token.value == ";"


def test_error():
    lexer = QasmLexer('θ')

    with pytest.raises(QasmException, match="Illegal character 'θ' at line 1"):
        lexer.token()
