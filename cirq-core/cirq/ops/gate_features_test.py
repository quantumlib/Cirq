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
from cirq.testing import assert_deprecated


def test_single_qubit_gate_validate_args():
    class Dummy(cirq.SingleQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')

    assert g.num_qubits() == 1
    g.validate_args([q1])
    g.validate_args([q2])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2])


def test_single_qubit_gate_validates_on_each():
    class Dummy(cirq.SingleQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    assert g.num_qubits() == 1

    test_qubits = [cirq.NamedQubit(str(i)) for i in range(3)]

    _ = g.on_each(*test_qubits)
    _ = g.on_each(test_qubits)

    test_non_qubits = [str(i) for i in range(3)]
    with pytest.raises(ValueError):
        _ = g.on_each(*test_non_qubits)
    with pytest.raises(ValueError):
        _ = g.on_each(*test_non_qubits)


def test_single_qubit_validates_on():
    class Dummy(cirq.SingleQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    assert g.num_qubits() == 1

    test_qubits = [cirq.NamedQubit(str(i)) for i in range(3)]

    with pytest.raises(ValueError):
        _ = g.on(*test_qubits)
    with pytest.raises(ValueError):
        _ = g.on(*test_qubits)


def test_two_qubit_gate_is_abstract_can_implement():
    class Included(cirq.TwoQubitGate):
        def matrix(self):
            pass

    with assert_deprecated(deadline="v0.14", count=2):
        assert isinstance(Included(), cirq.TwoQubitGate)


def test_two_qubit_gate_validate_pass():
    class Dummy(cirq.TwoQubitGate):
        def matrix(self):
            pass

    with assert_deprecated(deadline="v0.14"):
        g = Dummy()

    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')

    assert g.num_qubits() == 2
    g.validate_args([q1, q2])
    g.validate_args([q2, q3])
    g.validate_args([q3, q2])


def test_two_qubit_gate_validate_wrong_number():
    class Dummy(cirq.TwoQubitGate):
        def matrix(self):
            pass

    with assert_deprecated(deadline="v0.14"):
        g = Dummy()

    q1 = cirq.NamedQubit('q1')
    q2 = cirq.NamedQubit('q2')
    q3 = cirq.NamedQubit('q3')

    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2, q3])


def test_three_qubit_gate_validate():
    class Dummy(cirq.ThreeQubitGate):
        def matrix(self):
            pass

    with assert_deprecated(deadline="v0.14"):
        g = Dummy()

    a, b, c, d = cirq.LineQubit.range(4)

    assert g.num_qubits() == 3

    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])


def test_qasm_output_args_validate():
    args = cirq.QasmArgs(version='2.0')
    args.validate_version('2.0')

    with pytest.raises(ValueError):
        args.validate_version('2.1')


def test_qasm_output_args_format():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    m_a = cirq.measure(a, key='meas_a')
    m_b = cirq.measure(b, key='meas_b')
    args = cirq.QasmArgs(
        precision=4,
        version='2.0',
        qubit_id_map={a: 'aaa[0]', b: 'bbb[0]'},
        meas_key_id_map={'meas_a': 'm_a', 'meas_b': 'm_b'},
    )

    assert args.format('_{0}_', a) == '_aaa[0]_'
    assert args.format('_{0}_', b) == '_bbb[0]_'

    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_a)) == '_m_a_'
    assert args.format('_{0:meas}_', cirq.measurement_key_name(m_b)) == '_m_b_'

    assert args.format('_{0}_', 89.1234567) == '_89.1235_'
    assert args.format('_{0}_', 1.23) == '_1.23_'

    assert args.format('_{0:half_turns}_', 89.1234567) == '_pi*89.1235_'
    assert args.format('_{0:half_turns}_', 1.23) == '_pi*1.23_'

    assert args.format('_{0}_', 'other') == '_other_'


def test_multi_qubit_gate_validate():
    class Dummy(cirq.Gate):
        def num_qubits(self) -> int:
            return self._num_qubits

        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

    a, b, c, d = cirq.LineQubit.range(4)

    g = Dummy(3)

    assert g.num_qubits() == 3
    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])


def test_supports_on_each_inheritance_shim():
    class NotOnEach(cirq.Gate):
        def num_qubits(self):
            return 1  # coverage: ignore

    class OnEach(cirq.ops.gate_features.SupportsOnEachGate):
        def num_qubits(self):
            return 1  # coverage: ignore

    class SingleQ(cirq.SingleQubitGate):
        pass

    not_on_each = NotOnEach()
    single_q = SingleQ()
    two_q = cirq.testing.TwoQubitGate()
    with assert_deprecated(deadline="v0.14"):
        on_each = OnEach()

    assert not isinstance(not_on_each, cirq.ops.gate_features.SupportsOnEachGate)
    assert isinstance(on_each, cirq.ops.gate_features.SupportsOnEachGate)
    assert isinstance(single_q, cirq.ops.gate_features.SupportsOnEachGate)
    assert not isinstance(two_q, cirq.ops.gate_features.SupportsOnEachGate)
    assert isinstance(cirq.X, cirq.ops.gate_features.SupportsOnEachGate)
    assert not isinstance(cirq.CX, cirq.ops.gate_features.SupportsOnEachGate)
    assert isinstance(cirq.DepolarizingChannel(0.01), cirq.ops.gate_features.SupportsOnEachGate)


def test_supports_on_each_deprecation():
    class CustomGate(cirq.ops.gate_features.SupportsOnEachGate):
        def num_qubits(self):
            return 1  # coverage: ignore

    with assert_deprecated(deadline="v0.14"):
        assert isinstance(CustomGate(), cirq.ops.gate_features.SupportsOnEachGate)


def test_supports_two_qubit_inheritance_shim():
    print()

    class Dummy1(cirq.Gate):
        def num_qubits(self):
            return 1

    class Dummy1a(cirq.SingleQubitGate):
        pass

    class Dummy2(cirq.Gate):
        def num_qubits(self):
            return 2

    class Dummy2a(cirq.TwoQubitGate):
        pass

    class NottaGate:
        def _num_qubits_(self):
            return 2  # coverage: ignore

    g1 = Dummy1()
    g1a = Dummy1a()
    g2 = Dummy2()
    with assert_deprecated(deadline="v0.14"):
        g2a = Dummy2a()

    assert not isinstance(g1, cirq.TwoQubitGate)
    assert not isinstance(g1a, cirq.TwoQubitGate)
    assert isinstance(g2, cirq.TwoQubitGate)
    assert isinstance(g2a, cirq.TwoQubitGate)
    assert not isinstance(cirq.X, cirq.TwoQubitGate)
    assert isinstance(cirq.CX, cirq.TwoQubitGate)
    assert not isinstance(NottaGate(), cirq.TwoQubitGate)


def test_supports_three_qubit_inheritance_shim():
    print()

    class Dummy1(cirq.Gate):
        def num_qubits(self):
            return 1

    class Dummy1a(cirq.SingleQubitGate):
        pass

    class Dummy3(cirq.Gate):
        def num_qubits(self):
            return 3

    class Dummy3a(cirq.ThreeQubitGate):
        pass

    class NottaGate:
        def _num_qubits_(self):
            return 3  # coverage: ignore

    g1 = Dummy1()
    g1a = Dummy1a()
    g3 = Dummy3()
    with assert_deprecated(deadline="v0.14"):
        g3a = Dummy3a()

    assert not isinstance(g1, cirq.ThreeQubitGate)
    assert not isinstance(g1a, cirq.ThreeQubitGate)
    assert isinstance(g3, cirq.ThreeQubitGate)
    assert isinstance(g3a, cirq.ThreeQubitGate)
    assert not isinstance(cirq.X, cirq.ThreeQubitGate)
    assert isinstance(cirq.CCX, cirq.ThreeQubitGate)
    assert not isinstance(NottaGate(), cirq.ThreeQubitGate)
