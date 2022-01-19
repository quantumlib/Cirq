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

import cirq
from cirq.optimizers.deferred_measurements import defer_measurements


def test_diagram():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls('a'))

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
0: ─────@────────────────
        │
1: ─────┼───X────────────
        │   │
a 0: ───X───@───M('a')───
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_measurements():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
            ┌──┐
0: ─────@────@──────────────
        │    │
1: ─────┼────┼X─────────────
        │    ││
a 0: ───X────┼@────M('a')───
             │
b 0: ────────X─────M('b')───
            └──┘
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_controlled_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.CX(q0, q1).with_classical_controls('a'),
    )

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
0: ─────@───@────────────
        │   │
1: ─────┼───X────────────
        │   │
a 0: ───X───@───M('a')───
""",
        use_unicode_characters=True,
    )


def test_diagram_extra_control_bits():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q0, key='b'),
        cirq.X(q1).with_classical_controls('a', 'b'),
    )

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
0: ─────@───@────────────────
        │   │
1: ─────┼───┼───X────────────
        │   │   │
a 0: ───X───┼───@───M('a')───
            │   │
b 0: ───────X───@───M('b')───
""",
        use_unicode_characters=True,
    )


def test_diagram_multiple_ops_single_moment():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.measure(q0, key='a'),
        cirq.measure(q1, key='b'),
        cirq.X(q0).with_classical_controls('a'),
        cirq.X(q1).with_classical_controls('b'),
    )

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
        ┌──┐   ┌──┐
0: ──────@──────X──────────────
         │      │
1: ──────┼@─────┼X─────────────
         ││     ││
a 0: ────X┼─────@┼────M('a')───
          │      │
b 1: ─────X──────@────M('b')───
        └──┘   └──┘
""",
        use_unicode_characters=True,
    )


def test_diagram_subcircuit():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(
        cirq.CircuitOperation(
            cirq.FrozenCircuit(
                cirq.measure(q0, key='a'),
                cirq.X(q1).with_classical_controls('a'),
            )
        )
    )

    cirq.testing.assert_has_diagram(
        defer_measurements(circuit),
        """
        [ 0: ─────@─────── ]
        [         │        ]
0: ─────[ 1: ─────┼───X─── ]────────────
        [         │   │    ]
        [ a 0: ───X───@─── ]
        │
1: ─────#2──────────────────────────────
        │
a 0: ───#3─────────────────────M('a')───
""",
        use_unicode_characters=True,
    )


def test_scope_local():
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(
        cirq.measure(q, key='a'),
        cirq.X(q).with_classical_controls('a'),
    )
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    assert not cirq.control_keys(outer_subcircuit)
    cirq.testing.assert_has_diagram(
        defer_measurements(cirq.Circuit(outer_subcircuit)),
        """
            [           [ 0: ─────@───X─── ]             ]
            [ 0: ───────[         │   │    ]──────────── ]
            [           [ a 0: ───X───@─── ](loops=2)    ]
0: ─────────[           │                                ]─────────────────────────
            [ 0:a 0: ───#2────────────────────────────── ]
            [           │                                ]
            [ 1:a 0: ───#3────────────────────────────── ](loops=2)
            │
0:0:a 0: ───#2────────────────────────────────────────────────────────M('0:0:a')───
            │
0:1:a 0: ───#3────────────────────────────────────────────────────────M('0:1:a')───
            │
1:0:a 0: ───#4────────────────────────────────────────────────────────M('1:0:a')───
            │
1:1:a 0: ───#5────────────────────────────────────────────────────────M('1:1:a')───
""",
        use_unicode_characters=True,
    )
    circuit = cirq.Circuit(outer_subcircuit)
    s = defer_measurements(circuit).freeze()
    cirq.testing.assert_has_diagram(
        cirq.CircuitOperation(s).mapped_circuit(deep=True),
        """
0: ─────────@───X───@───X───@───X───@───X────────────────
            │   │   │   │   │   │   │   │
0:0:a 0: ───X───@───┼───┼───┼───┼───┼───┼───M('0:0:a')───
                    │   │   │   │   │   │
0:1:a 0: ───────────X───@───┼───┼───┼───┼───M('0:1:a')───
                            │   │   │   │
1:0:a 0: ───────────────────X───@───┼───┼───M('1:0:a')───
                                    │   │
1:1:a 0: ───────────────────────────X───@───M('1:1:a')───
""",
        use_unicode_characters=True,
    )
    assert circuit == cirq.Circuit(cirq.decompose(outer_subcircuit))
