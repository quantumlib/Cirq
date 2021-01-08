# Copyright 2019 The Cirq Developers
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

import numpy as np
import pytest

import cirq


def test_init():
    op = cirq.GlobalPhaseOperation(1j)
    assert op.coefficient == 1j
    assert op.qubits == ()
    assert op.with_qubits() is op

    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.GlobalPhaseOperation(2)
    with pytest.raises(ValueError, match='0 qubits'):
        _ = cirq.GlobalPhaseOperation(1j).with_qubits(cirq.LineQubit(0))


def test_protocols():
    for p in [1, 1j, -1]:
        cirq.testing.assert_implements_consistent_protocols(cirq.GlobalPhaseOperation(p))

    np.testing.assert_allclose(
        cirq.unitary(cirq.GlobalPhaseOperation(1j)), np.array([[1j]]), atol=1e-8
    )


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_act_on_tableau(phase):
    original_tableau = cirq.CliffordTableau(0)
    args = cirq.ActOnCliffordTableauArgs(original_tableau.copy(), [], np.random.RandomState(), {})
    cirq.act_on(cirq.GlobalPhaseOperation(phase), args, allow_decompose=False)
    assert args.tableau == original_tableau


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_act_on_ch_form(phase):
    state = cirq.StabilizerStateChForm(0)
    args = cirq.ActOnStabilizerCHFormArgs(
        state,
        [],
        prng=np.random.RandomState(),
        log_of_measurement_results={},
    )
    cirq.act_on(cirq.GlobalPhaseOperation(phase), args, allow_decompose=False)
    assert state.state_vector() == [[phase]]


def test_str():
    assert str(cirq.GlobalPhaseOperation(1j)) == '1j'


def test_repr():
    op = cirq.GlobalPhaseOperation(1j)
    cirq.testing.assert_equivalent_repr(op)


def test_diagram():
    a, b = cirq.LineQubit.range(2)
    x, y = cirq.LineQubit.range(10, 12)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.CNOT(a, x),
                        cirq.CNOT(b, y),
                        cirq.GlobalPhaseOperation(-1),
                    ]
                )
            ]
        ),
        """
                ┌──┐
0: ──────────────@─────
                 │
1: ──────────────┼@────
                 ││
10: ─────────────X┼────
                  │
11: ──────────────X────

global phase:    π
                └──┘
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.CNOT(a, x),
                        cirq.CNOT(b, y),
                        cirq.GlobalPhaseOperation(-1),
                        cirq.GlobalPhaseOperation(-1),
                    ]
                ),
            ]
        ),
        """
                ┌──┐
0: ──────────────@─────
                 │
1: ──────────────┼@────
                 ││
10: ─────────────X┼────
                  │
11: ──────────────X────

global phase:
                └──┘
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.CNOT(a, x),
                        cirq.CNOT(b, y),
                        cirq.GlobalPhaseOperation(-1),
                        cirq.GlobalPhaseOperation(-1),
                    ]
                ),
                cirq.Moment(
                    [
                        cirq.GlobalPhaseOperation(1j),
                    ]
                ),
                cirq.Moment(
                    [
                        cirq.X(a),
                    ]
                ),
            ]
        ),
        """
                ┌──┐
0: ──────────────@────────────X───
                 │
1: ──────────────┼@───────────────
                 ││
10: ─────────────X┼───────────────
                  │
11: ──────────────X───────────────

global phase:          0.5π
                └──┘
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.X(a),
                    ]
                ),
                cirq.Moment(
                    [
                        cirq.GlobalPhaseOperation(-1j),
                    ]
                ),
            ]
        ),
        """
0: ─────────────X───────────

global phase:       -0.5π
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.X(a),
                        cirq.GlobalPhaseOperation(np.exp(1j)),
                    ]
                ),
            ]
        ),
        """
0: ─────────────X────────

global phase:   0.318π
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.X(a),
                        cirq.GlobalPhaseOperation(np.exp(1j)),
                    ]
                ),
            ]
        ),
        """
0: ─────────────X──────────

global phase:   0.31831π
        """,
        precision=5,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.X(a),
                        cirq.GlobalPhaseOperation(1j),
                    ]
                ),
                cirq.Moment(
                    [
                        cirq.GlobalPhaseOperation(-1j),
                    ]
                ),
            ]
        ),
        """
0: -------------X----------------

global phase:   0.5pi   -0.5pi
        """,
        use_unicode_characters=False,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment(
                    [
                        cirq.GlobalPhaseOperation(-1j),
                    ]
                ),
            ]
        ),
        """
global phase:   -0.5π
        """,
    )


def test_global_phase_op_json_dict():
    assert cirq.GlobalPhaseOperation(-1j)._json_dict_() == {
        'cirq_type': 'GlobalPhaseOperation',
        'coefficient': -1j,
    }
