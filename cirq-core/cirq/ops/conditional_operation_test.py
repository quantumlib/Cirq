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
import sympy

import cirq
from cirq.ops.conditional_operation import ConditionalOperation


def test_diagram():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.measure(q0, key='a'), ConditionalOperation(cirq.X(q1), ['a']))

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───M───────
      ║
1: ───╫───X───
      ║   ║
a: ═══@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_pauli():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure_single_paulistring(cirq.X(q0), key='a'),
        ConditionalOperation(cirq.X(q1), ['a']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───M(X)───────
      ║
1: ───╫──────X───
      ║      ║
a: ═══@══════^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        ConditionalOperation(cirq.X(q1), ['a']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───M───M('b')───
      ║
1: ───╫───X────────
      ║   ║
a: ═══@═══^════════
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        ConditionalOperation(cirq.CX(q0, q1), ['a']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───M───@───
      ║   ║
1: ───╫───X───
      ║   ║
a: ═══@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        ConditionalOperation(cirq.X(q1), ['a', 'b']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
0: ───M───M───────
      ║   ║
1: ───╫───╫───X───
      ║   ║   ║
a: ═══@═══╬═══^═══
          ║   ║
b: ═══════@═══^═══
""",
        use_unicode_characters=True,
    )


def test_diagram_multiple_ops_single_moment():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        ConditionalOperation(cirq.X(q0), ['a']),
        ConditionalOperation(cirq.X(q1), ['b']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
      ┌──┐   ┌──┐
0: ────M──────X─────
       ║      ║
1: ────╫M─────╫X────
       ║║     ║║
a: ════@╬═════^╬════
        ║      ║
b: ═════@══════^════
      └──┘   └──┘
""",
        use_unicode_characters=True,
    )


def test_diagram_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                ConditionalOperation(cirq.X(q1), ['a']),
            )
        )
    )

    cirq.testing.assert_has_diagram(
        c,
        """
      Circuit_0x0000000000000000:
      [ 0: ───M───────          ]
0: ───[       ║                 ]───
      [ 1: ───╫───X───          ]
      [       ║   ║             ]
      [ a: ═══@═══^═══          ]
      │
1: ───#2────────────────────────────
""",
        use_unicode_characters=True,
    )


def test_diagram_subcircuit_layered():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                ConditionalOperation(cirq.X(q1), ['a']),
            ),
        ),
        ConditionalOperation(cirq.X(q1), ['a']),
    )

    cirq.testing.assert_has_diagram(
        c,
        """
          Circuit_0x0000000000000000:
          [ 0: ───M───────          ]
0: ───M───[       ║                 ]───────
      ║   [ 1: ───╫───X───          ]
      ║   [       ║   ║             ]
      ║   [ a: ═══@═══^═══          ]
      ║   ║
1: ───╫───#2────────────────────────────X───
      ║   ║                             ║
a: ═══@═══╩═════════════════════════════^═══
""",
        use_unicode_characters=True,
    )


def test_key_unset():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.measure(q0, key='a'),
        ConditionalOperation(cirq.X(q1), ['a']),
        cirq.measure(q1, key='b'),
    )
    s = cirq.Simulator()
    result = s.run(c)
    assert result.measurements['a'] == 0
    assert result.measurements['b'] == 0


def test_key_set():
    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='a'),
        ConditionalOperation(cirq.X(q1), ['a']),
        cirq.measure(q1, key='b'),
    )
    s = cirq.Simulator()
    result = s.run(c)
    assert result.measurements['a'] == 1
    assert result.measurements['b'] == 1


def test_subcircuit_key_unset():
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.Circuit(
        cirq.measure(q0, key='c'),
        ConditionalOperation(cirq.X(q1), ['c']),
        cirq.measure(q1, key='b'),
    )
    c = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=2, measurement_key_map={'c': 'a'})
    )
    s = cirq.Simulator()
    result = s.run(c)
    assert result.measurements['0:a'] == 0
    assert result.measurements['0:b'] == 0
    assert result.measurements['1:a'] == 0
    assert result.measurements['1:b'] == 0


def test_subcircuit_key_set():
    q0, q1 = cirq.LineQubit.range(2)
    inner = cirq.Circuit(
        cirq.X(q0),
        cirq.measure(q0, key='c'),
        ConditionalOperation(cirq.X(q1), ['c']),
        cirq.measure(q1, key='b'),
    )
    c = cirq.Circuit(
        cirq.CircuitOperation(inner.freeze(), repetitions=8, measurement_key_map={'c': 'a'})
    )
    s = cirq.Simulator()
    result = s.run(c)
    assert result.measurements['0:a'] == 1
    assert result.measurements['0:b'] == 1
    assert result.measurements['1:a'] == 0
    assert result.measurements['1:b'] == 1
    assert result.measurements['2:a'] == 1
    assert result.measurements['2:b'] == 0
    assert result.measurements['3:a'] == 0
    assert result.measurements['3:b'] == 0
    assert result.measurements['4:a'] == 1
    assert result.measurements['4:b'] == 1
    assert result.measurements['5:a'] == 0
    assert result.measurements['5:b'] == 1
    assert result.measurements['6:a'] == 1
    assert result.measurements['6:b'] == 0
    assert result.measurements['7:a'] == 0
    assert result.measurements['7:b'] == 0


def test_key_stacking():
    q0 = cirq.LineQubit(0)
    inner = cirq.X(q0)
    op = ConditionalOperation(ConditionalOperation(inner, ['a']), ['b'])
    assert op.sub_operation is inner
    assert set(map(str, op.controls)) == {'a', 'b'}


def test_qubit_mapping():
    q0, q1 = cirq.LineQubit.range(2)
    op = ConditionalOperation(cirq.X(q0), ['a'])
    assert op.with_qubits(q1).qubits == (q1,)


def test_parameterizable():
    a = sympy.Symbol('S')
    q0 = cirq.LineQubit(0)
    op = ConditionalOperation(cirq.X(q0), ['a'])
    opa = ConditionalOperation(cirq.XPowGate(exponent=a).on(q0), ['a'])
    assert cirq.is_parameterized(opa)
    assert not cirq.is_parameterized(op)
    assert cirq.resolve_parameters(opa, cirq.ParamResolver({'S': 1})) == op


def test_decompose():
    q0 = cirq.LineQubit(0)
    op = ConditionalOperation(cirq.H(q0), ['a'])
    assert cirq.decompose(op) == [
        ConditionalOperation(cirq.Y(q0) ** 0.5, ['a']),
        ConditionalOperation(cirq.XPowGate(exponent=1.0, global_shift=-0.25).on(q0), ['a']),
    ]


def test_str():
    q0 = cirq.LineQubit(0)
    op = ConditionalOperation(cirq.X(q0), ['a'])
    assert (
        str(op)
        == "ConditionalOperation(cirq.X(cirq.LineQubit(0)), [cirq.MeasurementKey(name='a')])"
    )


def test_pow():
    q0 = cirq.LineQubit(0)
    inner = cirq.X(q0)
    op = ConditionalOperation(inner, ['a']) ** 2
    assert op.sub_operation == inner ** 2
