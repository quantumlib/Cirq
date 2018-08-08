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
from cirq.ops import gate_features, raw_types, common_gates


def test_reversible_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ReversibleEffect()


def test_reversible_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.ReversibleEffect):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_reversible_gate_is_abstract_can_implement():
    class Included(gate_features.ReversibleEffect):
        def inverse(self):
            pass

    assert isinstance(Included(), gate_features.ReversibleEffect)


def test_extrapolatable_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ExtrapolatableEffect()


def test_extrapolatable_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.ExtrapolatableEffect):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_extrapolatable_gate_is_abstract_can_implement():
    class Included(gate_features.ExtrapolatableEffect):
        def extrapolate_effect(self, factor):
            pass

    assert isinstance(Included(), gate_features.ExtrapolatableEffect)


def test_composite_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.CompositeGate()


def test_composite_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class Missing(gate_features.CompositeGate):
        pass

    with pytest.raises(TypeError):
        _ = Missing()


def test_composite_gate_is_abstract_can_implement():
    class Included(gate_features.CompositeGate):
        def default_decompose(self, qubits):
            pass

    assert isinstance(Included(), gate_features.CompositeGate)


def test_single_qubit_gate_validate_args():
    class Dummy(gate_features.SingleQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()

    g.validate_args([q1])
    g.validate_args([q2])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2])


def test_two_qubit_gate_is_abstract_can_implement():
    class Included(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    assert isinstance(Included(),
                      gate_features.TwoQubitGate)


def test_two_qubit_gate_validate_pass():
    class Dummy(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()
    q3 = raw_types.QubitId()

    g.validate_args([q1, q2])
    g.validate_args([q2, q3])
    g.validate_args([q3, q2])


def test_two_qubit_gate_validate_wrong_number():
    class Dummy(gate_features.TwoQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    q1 = raw_types.QubitId()
    q2 = raw_types.QubitId()
    q3 = raw_types.QubitId()

    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([q1])
    with pytest.raises(ValueError):
        g.validate_args([q1, q2, q3])


def test_three_qubit_gate_validate():
    class Dummy(gate_features.ThreeQubitGate):
        def matrix(self):
            pass

    g = Dummy()
    a, b, c, d = cirq.LineQubit.range(4)

    g.validate_args([a, b, c])
    with pytest.raises(ValueError):
        g.validate_args([])
    with pytest.raises(ValueError):
        g.validate_args([a])
    with pytest.raises(ValueError):
        g.validate_args([a, b])
    with pytest.raises(ValueError):
        g.validate_args([a, b, c, d])


def test_parameterizable_gate_is_abstract_cant_instantiate():
    with pytest.raises(TypeError):
        _ = gate_features.ParameterizableEffect()


def test_parameterizable_gate_is_abstract_must_implement():
    # noinspection PyAbstractClass
    class MissingBoth(gate_features.ParameterizableEffect):
        pass
    # noinspection PyAbstractClass
    class MissingOne(gate_features.ParameterizableEffect):
        def is_parameterized(self):
            pass
    # noinspection PyAbstractClass
    class MissingOtherOne(gate_features.ParameterizableEffect):
        def with_parameters_resolved_by(self, param_resolver):
            pass

    with pytest.raises(TypeError):
        _ = MissingBoth()
    with pytest.raises(TypeError):
        _ = MissingOne()
    with pytest.raises(TypeError):
        _ = MissingOtherOne()


def test_parameterizable_gate_is_abstract_can_implement():
    class Included(gate_features.ParameterizableEffect):
        def is_parameterized(self):
            pass

        def with_parameters_resolved_by(self, param_resolver):
            pass

    assert isinstance(Included(), gate_features.ParameterizableEffect)


def test_on_each():
    class CustomGate(gate_features.SingleQubitGate):
        pass
    a = raw_types.NamedQubit('a')
    b = raw_types.NamedQubit('b')
    c = CustomGate()

    assert c.on_each([]) == []
    assert c.on_each([a]) == [c(a)]
    assert c.on_each([a, b]) == [c(a), c(b)]
    assert c.on_each([b, a]) == [c(b), c(a)]


@cirq.testing.only_test_in_python3
def test_text_diagram_info_repr():
    info = cirq.TextDiagramInfo(('X', 'Y'), 2)
    assert repr(info) == ("TextDiagramInfo(wire_symbols=('X', 'Y')"
                          ", exponent=2, connected=True)")


def test_text_diagram_info_eq():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: cirq.TextDiagramInfo(('X',)))
    eq.add_equality_group(cirq.TextDiagramInfo(('X', 'Y')),
                          cirq.TextDiagramInfo(('X', 'Y'), 1))
    eq.add_equality_group(cirq.TextDiagramInfo(('Z',), 2))
    eq.add_equality_group(cirq.TextDiagramInfo(('Z', 'Z'), 2))
    eq.add_equality_group(cirq.TextDiagramInfo(('Z',), 3))


def test_qasm_output_args_validate():
    args = cirq.QasmOutputArgs(version='2.0')
    args.validate_version('2.0')

    with pytest.raises(ValueError):
        args.validate_version('2.1')


def test_qasm_output_args_format():
    a = raw_types.NamedQubit('a')
    b = raw_types.NamedQubit('b')
    m_a = common_gates.MeasurementGate('meas_a')(a)
    m_b = common_gates.MeasurementGate('meas_b')(b)
    args = cirq.QasmOutputArgs(
                    precision=4,
                    version='2.0',
                    qubit_id_map={a: 'aaa[0]', b: 'bbb[0]'},
                    meas_key_id_map={'meas_a': 'm_a', 'meas_b': 'm_b'})

    assert args.format('_{0}_', a) == '_aaa[0]_'
    assert args.format('_{0}_', b) == '_bbb[0]_'

    assert args.format('_{0:meas}_', m_a.gate.key) == '_m_a_'
    assert args.format('_{0:meas}_', m_b.gate.key) == '_m_b_'

    assert args.format('_{0}_', 89.1234567) == '_89.1235_'
    assert args.format('_{0}_', 1.23) == '_1.23_'

    assert args.format('_{0:half_turns}_', 89.1234567) == '_pi*89.1235_'
    assert args.format('_{0:half_turns}_', 1.23) == '_pi*1.23_'

    assert args.format('_{0}_', 'other') == '_other_'
