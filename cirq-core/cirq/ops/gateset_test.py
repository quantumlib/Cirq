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

from typing import Tuple, List, cast
import re
import pytest
import sympy
import cirq
from cirq._compat import proper_repr
import numpy as np


class CustomXPowGate(cirq.EigenGate):
    def _eigen_components(self) -> List[Tuple[float, np.ndarray]]:
        return [
            (0, np.array([[0.5, 0.5], [0.5, 0.5]])),
            (1, np.array([[0.5, -0.5], [-0.5, 0.5]])),
        ]

    def __str__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'CustomX'
            return f'CustomX**{self._exponent}'
        return f'CustomXPowGate(exponent={self._exponent}, global_shift={self._global_shift!r})'

    def __repr__(self) -> str:
        if self._global_shift == 0:
            if self._exponent == 1:
                return 'cirq.ops.gateset_test.CustomX'
            return f'(cirq.ops.gateset_test.CustomX**{proper_repr(self._exponent)})'
        return 'cirq.ops.gateset_test.CustomXPowGate(exponent={}, global_shift={!r})'.format(
            proper_repr(self._exponent), self._global_shift
        )

    def _num_qubits_(self) -> int:
        return 1


CustomX = CustomXPowGate()


@pytest.mark.parametrize('gate', [CustomX, CustomXPowGate])
def test_gate_family_init(gate):
    name = 'test_name'
    description = 'test_description'
    g = cirq.GateFamily(gate=gate, name=name, description=description)
    assert g.gate == gate
    assert g.name == name
    assert g.description == description


@pytest.mark.parametrize('gate', [CustomX, CustomXPowGate])
def test_gate_family_default_name_and_description(gate):
    g = cirq.GateFamily(gate)
    assert re.match('.*GateFamily.*CustomX.*', g.name)
    assert re.match('Accepts.*instances.*CustomX.*', g.description)


def test_invalid_gate_family():
    with pytest.raises(ValueError, match='instance or subclass of `cirq.Gate`'):
        _ = cirq.GateFamily(gate=cirq.Operation)

    with pytest.raises(ValueError, match='non-parameterized instance of `cirq.Gate`'):
        _ = cirq.GateFamily(gate=CustomX ** sympy.Symbol('theta'))


def test_gate_family_immutable():
    g = cirq.GateFamily(CustomX)
    with pytest.raises(AttributeError, match="can't set attribute"):
        g.gate = CustomXPowGate
    with pytest.raises(AttributeError, match="can't set attribute"):
        g.name = 'new name'
    with pytest.raises(AttributeError, match="can't set attribute"):
        g.description = 'new description'


@pytest.mark.parametrize(
    'gate', [CustomX, CustomXPowGate(exponent=0.5, global_shift=0.1), CustomXPowGate]
)
@pytest.mark.parametrize('name, description', [(None, None), ('custom_name', 'custom_description')])
def test_gate_family_repr_and_str(gate, name, description):
    g = cirq.GateFamily(gate, name=name, description=description)
    cirq.testing.assert_equivalent_repr(g)
    assert g.name in str(g)
    assert g.description in str(g)


@pytest.mark.parametrize('gate', [cirq.X, cirq.XPowGate(), cirq.XPowGate])
@pytest.mark.parametrize('name, description', [(None, None), ('custom_name', 'custom_description')])
def test_gate_family_json(gate, name, description):
    g = cirq.GateFamily(gate, name=name, description=description)
    g_json = cirq.to_json(g)
    assert cirq.read_json(json_text=g_json) == g


def test_gate_family_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.GateFamily(CustomX))
    eq.add_equality_group(cirq.GateFamily(CustomX ** 3))
    eq.add_equality_group(
        cirq.GateFamily(CustomX, name='custom_name', description='custom_description'),
        cirq.GateFamily(CustomX ** 3, name='custom_name', description='custom_description'),
    )
    eq.add_equality_group(cirq.GateFamily(CustomXPowGate))
    eq.add_equality_group(
        cirq.GateFamily(CustomXPowGate, name='custom_name', description='custom_description')
    )


@pytest.mark.parametrize(
    'gate_family, gates_to_check',
    [
        (
            cirq.GateFamily(CustomXPowGate),
            [
                (CustomX, True),
                (CustomX ** 0.5, True),
                (CustomX ** sympy.Symbol('theta'), True),
                (CustomXPowGate(exponent=0.25, global_shift=0.15), True),
                (cirq.SingleQubitGate(), False),
                (cirq.X ** 0.5, False),
                (None, False),
                (cirq.global_phase_operation(1j), False),
            ],
        ),
        (
            cirq.GateFamily(CustomX),
            [
                (CustomX, True),
                (CustomXPowGate(exponent=1, global_shift=0.15), True),
                (CustomX ** 2, False),
                (CustomX ** 3, True),
                (CustomX ** sympy.Symbol('theta'), False),
                (None, False),
                (cirq.global_phase_operation(1j), False),
            ],
        ),
        (
            cirq.GateFamily(CustomX, ignore_global_phase=False),
            [
                (CustomX, True),
                (CustomXPowGate(exponent=1, global_shift=0.15), False),
            ],
        ),
    ],
)
def test_gate_family_predicate_and_containment(gate_family, gates_to_check):
    q = cirq.NamedQubit("q")
    for gate, result in gates_to_check:
        assert gate_family._predicate(gate) == result
        assert (gate in gate_family) == result
        if isinstance(gate, cirq.Gate):
            assert (gate(q) in gate_family) == result
            assert (gate(q).with_tags('tags') in gate_family) == result


class CustomXGateFamily(cirq.GateFamily):
    """Accepts all integer powers of CustomXPowGate"""

    def __init__(self) -> None:
        super().__init__(
            gate=CustomXPowGate,
            name='CustomXGateFamily',
            description='Accepts all integer powers of CustomXPowGate',
        )

    def _predicate(self, g: cirq.Gate) -> bool:
        """Checks whether gate instance `g` belongs to this GateFamily."""
        if not super()._predicate(g) or cirq.is_parameterized(g):
            return False
        exp = cast(CustomXPowGate, g).exponent
        return int(exp) == exp

    def __repr__(self):
        return 'cirq.ops.gateset_test.CustomXGateFamily()'


gateset = cirq.Gateset(
    CustomX ** 0.5, cirq.testing.TwoQubitGate, CustomXGateFamily(), name='custom gateset'
)


def test_gateset_init():
    assert gateset.name == 'custom gateset'
    assert gateset.gates == frozenset(
        [
            cirq.GateFamily(CustomX ** 0.5),
            cirq.GateFamily(cirq.testing.TwoQubitGate),
            CustomXGateFamily(),
        ]
    )


def test_gateset_repr_and_str():
    cirq.testing.assert_equivalent_repr(gateset)
    assert gateset.name in str(gateset)
    for gate_family in gateset.gates:
        assert str(gate_family) in str(gateset)


@pytest.mark.parametrize(
    'gate, result',
    [
        (CustomX, True),
        (CustomX ** 2, True),
        (CustomXPowGate(exponent=3, global_shift=0.5), True),
        (CustomX ** 0.5, True),
        (CustomXPowGate(exponent=0.5, global_shift=0.5), True),
        (CustomX ** 0.25, False),
        (CustomX ** sympy.Symbol('theta'), False),
        (cirq.testing.TwoQubitGate(), True),
    ],
)
def test_gateset_contains(gate, result):
    assert (gate in gateset) is result
    op = gate(*cirq.LineQubit.range(gate.num_qubits()))
    assert (op in gateset) is result
    assert (op.with_tags('tags') in gateset) is result
    circuit_op = cirq.CircuitOperation(cirq.FrozenCircuit([op] * 5), repetitions=5)
    assert (circuit_op in gateset) is result
    assert circuit_op not in gateset.with_params(unroll_circuit_op=False)


@pytest.mark.parametrize('use_circuit_op', [True, False])
@pytest.mark.parametrize('use_global_phase', [True, False])
def test_gateset_validate(use_circuit_op, use_global_phase):
    def optree_and_circuit(optree):
        yield optree
        yield cirq.Circuit(optree)

    def get_ops(use_circuit_op, use_global_phase):
        q = cirq.LineQubit.range(3)
        yield [CustomX(q[0]).with_tags('custom tags'), CustomX(q[1]) ** 2, CustomX(q[2]) ** 3]
        yield [CustomX(q[0]) ** 0.5, cirq.testing.TwoQubitGate()(*q[:2])]
        if use_circuit_op:
            circuit_op = cirq.CircuitOperation(
                cirq.FrozenCircuit(get_ops(False, False)), repetitions=10
            ).with_tags('circuit op tags')
            recursive_circuit_op = cirq.CircuitOperation(
                cirq.FrozenCircuit([circuit_op, CustomX(q[2]) ** 0.5]),
                repetitions=10,
                qubit_map={q[0]: q[1], q[1]: q[2], q[2]: q[0]},
            )
            yield [circuit_op, recursive_circuit_op]
        if use_global_phase:
            yield cirq.global_phase_operation(1j)

    def assert_validate_and_contains_consistent(gateset, op_tree, result):
        assert all(op in gateset for op in cirq.flatten_to_ops(op_tree)) is result
        for item in optree_and_circuit(op_tree):
            assert gateset.validate(item) is result

    op_tree = [*get_ops(use_circuit_op, use_global_phase)]
    assert_validate_and_contains_consistent(
        gateset.with_params(
            unroll_circuit_op=use_circuit_op,
            accept_global_phase_op=use_global_phase,
        ),
        op_tree,
        True,
    )
    if use_circuit_op or use_global_phase:
        assert_validate_and_contains_consistent(
            gateset.with_params(
                unroll_circuit_op=False,
                accept_global_phase_op=False,
            ),
            op_tree,
            False,
        )


def test_gateset_validate_circuit_op_negative_reps():
    gate = CustomXPowGate(exponent=0.5)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(gate.on(cirq.LineQubit(0))), repetitions=-1)
    assert op not in cirq.Gateset(gate)
    assert op ** -1 in cirq.Gateset(gate)


def test_with_params():
    assert gateset.with_params() is gateset
    assert (
        gateset.with_params(
            name=gateset.name,
            unroll_circuit_op=gateset._unroll_circuit_op,
            accept_global_phase_op=gateset._accept_global_phase_op,
        )
        is gateset
    )
    gateset_with_params = gateset.with_params(
        name='new name', unroll_circuit_op=False, accept_global_phase_op=False
    )
    assert gateset_with_params.name == 'new name'
    assert gateset_with_params._unroll_circuit_op is False
    assert gateset_with_params._accept_global_phase_op is False


def test_gateset_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.Gateset(CustomX))
    eq.add_equality_group(cirq.Gateset(CustomX ** 3))
    eq.add_equality_group(cirq.Gateset(CustomX, name='Custom Gateset'))
    eq.add_equality_group(cirq.Gateset(CustomX, name='Custom Gateset', unroll_circuit_op=False))
    eq.add_equality_group(
        cirq.Gateset(CustomX, name='Custom Gateset', accept_global_phase_op=False)
    )
    eq.add_equality_group(
        cirq.Gateset(
            cirq.GateFamily(CustomX, name='custom_name', description='custom_description'),
            cirq.GateFamily(CustomX, name='custom_name', description='custom_description'),
        ),
        cirq.Gateset(
            cirq.GateFamily(CustomX ** 3, name='custom_name', description='custom_description'),
            cirq.GateFamily(CustomX, name='custom_name', description='custom_description'),
        ),
    )
    eq.add_equality_group(
        cirq.Gateset(CustomX, CustomXPowGate),
        cirq.Gateset(CustomXPowGate, CustomX),
        cirq.Gateset(CustomX, CustomX, CustomXPowGate),
        cirq.Gateset(CustomXPowGate, CustomX, CustomXPowGate),
    )
    eq.add_equality_group(cirq.Gateset(CustomXGateFamily()))
    eq.add_equality_group(
        cirq.Gateset(
            cirq.GateFamily(
                gate=CustomXPowGate,
                name='CustomXGateFamily',
                description='Accepts all integer powers of CustomXPowGate',
            )
        )
    )
