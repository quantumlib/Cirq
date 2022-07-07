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
import sympy

import cirq


def test_init():
    op = cirq.global_phase_operation(1j)
    assert op.gate.coefficient == 1j
    assert op.qubits == ()
    assert op.with_qubits() == op
    assert cirq.has_stabilizer_effect(op)

    with pytest.raises(ValueError, match='not unitary'):
        _ = cirq.global_phase_operation(2)
    with pytest.raises(ValueError, match='0 qubits'):
        _ = cirq.global_phase_operation(1j).with_qubits(cirq.LineQubit(0))


def test_protocols():
    for p in [1, 1j, -1]:
        cirq.testing.assert_implements_consistent_protocols(cirq.global_phase_operation(p))

    np.testing.assert_allclose(
        cirq.unitary(cirq.global_phase_operation(1j)), np.array([[1j]]), atol=1e-8
    )


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_act_on_tableau(phase):
    original_tableau = cirq.CliffordTableau(0)
    args = cirq.CliffordTableauSimulationState(original_tableau.copy(), np.random.RandomState())
    cirq.act_on(cirq.global_phase_operation(phase), args, allow_decompose=False)
    assert args.tableau == original_tableau


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_act_on_ch_form(phase):
    state = cirq.StabilizerStateChForm(0)
    args = cirq.StabilizerChFormSimulationState(
        qubits=[], prng=np.random.RandomState(), initial_state=state
    )
    cirq.act_on(cirq.global_phase_operation(phase), args, allow_decompose=False)
    assert state.state_vector() == [[phase]]


def test_str():
    assert str(cirq.global_phase_operation(1j)) == '1j'


def test_repr():
    op = cirq.global_phase_operation(1j)
    cirq.testing.assert_equivalent_repr(op)


def test_diagram():
    a, b = cirq.LineQubit.range(2)
    x, y = cirq.LineQubit.range(10, 12)

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [cirq.Moment([cirq.CNOT(a, x), cirq.CNOT(b, y), cirq.global_phase_operation(-1)])]
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
                        cirq.global_phase_operation(-1),
                        cirq.global_phase_operation(-1),
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
                        cirq.global_phase_operation(-1),
                        cirq.global_phase_operation(-1),
                    ]
                ),
                cirq.Moment([cirq.global_phase_operation(1j)]),
                cirq.Moment([cirq.X(a)]),
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
        cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.global_phase_operation(-1j)])]),
        """
0: ─────────────X───────────

global phase:       -0.5π
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit([cirq.Moment([cirq.X(a), cirq.global_phase_operation(np.exp(1j))])]),
        """
0: ─────────────X────────

global phase:   0.318π
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit([cirq.Moment([cirq.X(a), cirq.global_phase_operation(np.exp(1j))])]),
        """
0: ─────────────X──────────

global phase:   0.31831π
        """,
        precision=5,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            [
                cirq.Moment([cirq.X(a), cirq.global_phase_operation(1j)]),
                cirq.Moment([cirq.global_phase_operation(-1j)]),
            ]
        ),
        """
0: -------------X----------------

global phase:   0.5pi   -0.5pi
        """,
        use_unicode_characters=False,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit([cirq.Moment([cirq.global_phase_operation(-1j)])]),
        """
global phase:   -0.5π
        """,
    )


def test_gate_init():
    gate = cirq.GlobalPhaseGate(1j)
    assert gate.coefficient == 1j
    assert isinstance(gate.on(), cirq.GateOperation)
    assert gate.on().gate == gate
    assert cirq.has_stabilizer_effect(gate)

    with pytest.raises(ValueError, match='Coefficient is not unitary'):
        _ = cirq.GlobalPhaseGate(2)
    with pytest.raises(ValueError, match='Wrong number of qubits'):
        _ = gate.on(cirq.LineQubit(0))


def test_gate_protocols():
    for p in [1, 1j, -1]:
        cirq.testing.assert_implements_consistent_protocols(cirq.GlobalPhaseGate(p))

    np.testing.assert_allclose(cirq.unitary(cirq.GlobalPhaseGate(1j)), np.array([[1j]]), atol=1e-8)


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_gate_act_on_tableau(phase):
    original_tableau = cirq.CliffordTableau(0)
    args = cirq.CliffordTableauSimulationState(original_tableau.copy(), np.random.RandomState())
    cirq.act_on(cirq.GlobalPhaseGate(phase), args, qubits=(), allow_decompose=False)
    assert args.tableau == original_tableau


@pytest.mark.parametrize('phase', [1, 1j, -1])
def test_gate_act_on_ch_form(phase):
    state = cirq.StabilizerStateChForm(0)
    args = cirq.StabilizerChFormSimulationState(
        qubits=[], prng=np.random.RandomState(), initial_state=state
    )
    cirq.act_on(cirq.GlobalPhaseGate(phase), args, qubits=(), allow_decompose=False)
    assert state.state_vector() == [[phase]]


def test_gate_str():
    assert str(cirq.GlobalPhaseGate(1j)) == '1j'


def test_gate_repr():
    gate = cirq.GlobalPhaseGate(1j)
    cirq.testing.assert_equivalent_repr(gate)


def test_gate_op_repr():
    gate = cirq.GlobalPhaseGate(1j)
    cirq.testing.assert_equivalent_repr(gate.on())


def test_gate_global_phase_op_json_dict():
    assert cirq.GlobalPhaseGate(-1j)._json_dict_() == {'coefficient': -1j}


def test_parameterization():
    t = sympy.Symbol('t')
    gpt = cirq.GlobalPhaseGate(coefficient=t)
    assert cirq.is_parameterized(gpt)
    assert cirq.parameter_names(gpt) == {'t'}
    assert not cirq.has_unitary(gpt)
    assert gpt.coefficient == t
    assert (gpt**2).coefficient == t**2


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve(resolve_fn):
    t = sympy.Symbol('t')
    gpt = cirq.GlobalPhaseGate(coefficient=t)
    assert resolve_fn(gpt, {'t': -1}) == cirq.GlobalPhaseGate(coefficient=-1)


@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_error(resolve_fn):
    t = sympy.Symbol('t')
    gpt = cirq.GlobalPhaseGate(coefficient=t)
    with pytest.raises(ValueError, match='Coefficient is not unitary'):
        resolve_fn(gpt, {'t': -2})
