#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cirq.contrib.qasm_import._lexer import QasmLexer


def test_empty_circuit():
    assert QasmLexer("").token() is None


def test_numbers():
    assert str(QasmLexer("3").token()) == 'LexToken(NUMBER,3,1,0)'
    assert str(QasmLexer("03").token()) == 'LexToken(NUMBER,3,1,0)'
    assert str(QasmLexer("046").token()) == 'LexToken(NUMBER,46,1,0)'


def test_format():
    assert str(QasmLexer("OPENQASM 2.0;").token()) == \
           "LexToken(QASM20,'OPENQASM 2.0;',1,0)"
