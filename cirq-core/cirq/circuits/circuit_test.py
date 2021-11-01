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
import os
from collections import defaultdict
from random import randint, random, sample, randrange
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing
from cirq import circuits
from cirq import ops
from cirq.testing.devices import ValidatingTestDevice


class _Foxy(ValidatingTestDevice):
    def can_add_operation_into_moment(
        self, operation: 'ops.Operation', moment: 'ops.Moment'
    ) -> bool:
        if not super().can_add_operation_into_moment(operation, moment):
            return False
        # a fake rule for ensuring that no two CZs are executed at the same moment.
        # this will ensure that CZs are always in separate moments in this device
        return not (
            isinstance(operation.gate, ops.CZPowGate)
            and any(isinstance(op.gate, ops.CZPowGate) for op in moment.operations)
        )


FOXY = _Foxy(
    allowed_qubit_types=(cirq.GridQubit,),
    allowed_gates=(
        ops.CZPowGate,
        ops.XPowGate,
        ops.YPowGate,
        ops.ZPowGate,
    ),
    qubits=set(cirq.GridQubit.rect(2, 7)),
    name=f'{__name__}.FOXY',
    auto_decompose_gates=(ops.CCXPowGate,),
    validate_locality=True,
)


BCONE = ValidatingTestDevice(
    allowed_qubit_types=(cirq.GridQubit,),
    allowed_gates=(ops.XPowGate,),
    qubits={
        cirq.GridQubit(0, 6),
    },
    name=f'{__name__}.BCONE',
)


if TYPE_CHECKING:
    import cirq


class _MomentAndOpTypeValidatingDeviceType(cirq.Device):
    def validate_operation(self, operation):
        if not isinstance(operation, cirq.Operation):
            raise ValueError(f'not isinstance({operation!r}, {cirq.Operation!r})')

    def validate_moment(self, moment):
        if not isinstance(moment, cirq.Moment):
            raise ValueError(f'not isinstance({moment!r}, {cirq.Moment!r})')


moment_and_op_type_validating_device = _MomentAndOpTypeValidatingDeviceType()


class ControlOp(cirq.Operation):
    def __init__(self, keys):
        self._keys = keys

    def with_qubits(self, *new_qids):
        pass  # coverage: ignore

    @property
    def qubits(self):
        return []  # coverage: ignore

    def _control_key_names_(self):
        return self._keys


def test_alignment():
    assert repr(cirq.Alignment.LEFT) == 'cirq.Alignment.LEFT'
    assert repr(cirq.Alignment.RIGHT) == 'cirq.Alignment.RIGHT'


def test_insert_moment_types():
    x = cirq.NamedQubit('x')

    with pytest.raises(ValueError):
        moment_and_op_type_validating_device.validate_operation(cirq.Moment())

    with pytest.raises(ValueError):
        moment_and_op_type_validating_device.validate_moment(cirq.X(x))

    circuit = cirq.Circuit(device=moment_and_op_type_validating_device)

    moment_or_operation_tree = [cirq.X(x), cirq.Moment([cirq.Y(x)])]
    circuit.insert(0, moment_or_operation_tree)

    moment_or_operation_tree = [[cirq.Moment([cirq.X(x)])]]
    circuit.insert(0, moment_or_operation_tree)


def test_setitem():
    circuit = cirq.Circuit([cirq.Moment(), cirq.Moment()])

    circuit[1] = cirq.Moment([cirq.X(cirq.LineQubit(0))])
    assert circuit == cirq.Circuit([cirq.Moment(), cirq.Moment([cirq.X(cirq.LineQubit(0))])])

    circuit[1:1] = (
        cirq.Moment([cirq.Y(cirq.LineQubit(0))]),
        cirq.Moment([cirq.Z(cirq.LineQubit(0))]),
    )
    assert circuit == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.Y(cirq.LineQubit(0))]),
            cirq.Moment([cirq.Z(cirq.LineQubit(0))]),
            cirq.Moment([cirq.X(cirq.LineQubit(0))]),
        ]
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_equality(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq = cirq.testing.EqualsTester()

    # Default is empty. Iterables get listed.
    eq.add_equality_group(
        circuit_cls(),
        circuit_cls(device=cirq.UNCONSTRAINED_DEVICE),
        circuit_cls([]),
        circuit_cls(()),
    )
    eq.add_equality_group(circuit_cls([cirq.Moment()]), circuit_cls((cirq.Moment(),)))
    eq.add_equality_group(circuit_cls(device=FOXY))

    # Equality depends on structure and contents.
    eq.add_equality_group(circuit_cls([cirq.Moment([cirq.X(a)])]))
    eq.add_equality_group(circuit_cls([cirq.Moment([cirq.X(b)])]))
    eq.add_equality_group(circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)])]))
    eq.add_equality_group(circuit_cls([cirq.Moment([cirq.X(a), cirq.X(b)])]))

    # Big case.
    eq.add_equality_group(
        circuit_cls(
            [
                cirq.Moment([cirq.H(a), cirq.H(b)]),
                cirq.Moment([cirq.CZ(a, b)]),
                cirq.Moment([cirq.H(b)]),
            ]
        )
    )
    eq.add_equality_group(
        circuit_cls(
            [
                cirq.Moment([cirq.H(a)]),
                cirq.Moment([cirq.CNOT(a, b)]),
            ]
        )
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_approx_eq(circuit_cls):
    class TestDevice(cirq.Device):
        def validate_operation(self, operation: cirq.Operation) -> None:
            pass

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert not cirq.approx_eq(circuit_cls([cirq.Moment([cirq.X(a)])]), cirq.Moment([cirq.X(a)]))

    assert cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.X(a)])]), circuit_cls([cirq.Moment([cirq.X(a)])])
    )
    assert not cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.X(a)])]), circuit_cls([cirq.Moment([cirq.X(b)])])
    )

    assert cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=1e-9)(a)])]),
    )

    assert not cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)])]),
    )
    assert cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=0)(a)])]),
        circuit_cls([cirq.Moment([cirq.XPowGate(exponent=1e-7)(a)])]),
        atol=1e-6,
    )

    assert not cirq.approx_eq(
        circuit_cls([cirq.Moment([cirq.X(a)])]),
        circuit_cls([cirq.Moment([cirq.X(a)])], device=TestDevice()),
    )


def test_append_single():
    a = cirq.NamedQubit('a')

    c = cirq.Circuit()
    c.append(())
    assert c == cirq.Circuit()

    c = cirq.Circuit()
    c.append(cirq.X(a))
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)])])

    c = cirq.Circuit()
    c.append([cirq.X(a)])
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)])])


def test_append_control_key():
    q = cirq.LineQubit(0)

    c = cirq.Circuit()
    c.append(cirq.measure(q, key='a'))
    c.append(ControlOp(['a']))
    assert len(c) == 2

    c = cirq.Circuit()
    c.append(cirq.measure(q, key='a'))
    c.append(ControlOp(['b']))
    c.append(ControlOp(['b']))
    assert len(c) == 1


def test_append_multiple():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(b)])])

    c = cirq.Circuit()
    c.append([cirq.X(a), cirq.X(b)], cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    c = cirq.Circuit()
    c.append(cirq.X(a), cirq.InsertStrategy.EARLIEST)
    c.append(cirq.X(b), cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )


def test_append_moments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.Circuit()
    c.append(cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    c = cirq.Circuit()
    c.append(
        [cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.X(a), cirq.X(b)])],
        cirq.InsertStrategy.NEW,
    )
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_add_op_tree(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls()
    assert c + [cirq.X(a), cirq.Y(b)] == circuit_cls(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
        ]
    )

    assert c + cirq.X(a) == circuit_cls(cirq.X(a))
    assert c + [cirq.X(a)] == circuit_cls(cirq.X(a))
    assert c + [[[cirq.X(a)], []]] == circuit_cls(cirq.X(a))
    assert c + (cirq.X(a),) == circuit_cls(cirq.X(a))
    assert c + (cirq.X(a) for _ in range(1)) == circuit_cls(cirq.X(a))
    with pytest.raises(TypeError):
        _ = c + cirq.X


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_radd_op_tree(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls()
    assert [cirq.X(a), cirq.Y(b)] + c == circuit_cls(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
        ]
    )

    assert cirq.X(a) + c == circuit_cls(cirq.X(a))
    assert [cirq.X(a)] + c == circuit_cls(cirq.X(a))
    assert [[[cirq.X(a)], []]] + c == circuit_cls(cirq.X(a))
    assert (cirq.X(a),) + c == circuit_cls(cirq.X(a))
    assert (cirq.X(a) for _ in range(1)) + c == circuit_cls(cirq.X(a))
    with pytest.raises(AttributeError):
        _ = cirq.X + c
    with pytest.raises(TypeError):
        _ = 0 + c

    # non-empty circuit addition
    if circuit_cls == cirq.FrozenCircuit:
        d = cirq.FrozenCircuit(cirq.Y(b))
    else:
        d = cirq.Circuit()
        d.append(cirq.Y(b))
    assert [cirq.X(a)] + d == circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)])])
    assert cirq.Moment([cirq.X(a)]) + d == circuit_cls(
        [cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.Y(b)])]
    )

    # Preserves device.
    c = circuit_cls(device=FOXY)
    c2 = [] + c
    assert c2.device is FOXY
    assert c2 == c

    # Validates versus device.
    c = circuit_cls(device=FOXY)
    with pytest.raises(ValueError, match='Unsupported qubit'):
        _ = [cirq.X(cirq.NamedQubit('a'))] + c


def test_add_iadd_equivalence():
    q0, q1 = cirq.LineQubit.range(2)
    iadd_circuit = cirq.Circuit(cirq.X(q0))
    iadd_circuit += cirq.H(q1)

    add_circuit = cirq.Circuit(cirq.X(q0)) + cirq.H(q1)
    assert iadd_circuit == add_circuit


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_bool(circuit_cls):
    assert not circuit_cls()
    assert circuit_cls(cirq.X(cirq.NamedQubit('a')))


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_repr(circuit_cls):
    assert repr(circuit_cls()) == f'cirq.{circuit_cls.__name__}()'

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(
        [
            cirq.Moment([cirq.H(a), cirq.H(b)]),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(a, b)]),
        ]
    )
    cirq.testing.assert_equivalent_repr(c)
    assert (
        repr(c)
        == f"""cirq.{circuit_cls.__name__}([
    cirq.Moment(
        cirq.H(cirq.NamedQubit('a')),
        cirq.H(cirq.NamedQubit('b')),
    ),
    cirq.Moment(),
    cirq.Moment(
        cirq.CZ(cirq.NamedQubit('a'), cirq.NamedQubit('b')),
    ),
])"""
    )

    c = circuit_cls(device=FOXY)
    cirq.testing.assert_equivalent_repr(c)
    assert repr(c) == f'cirq.{circuit_cls.__name__}(device={repr(FOXY)})'

    c = circuit_cls(cirq.Z(cirq.GridQubit(0, 0)), device=FOXY)
    cirq.testing.assert_equivalent_repr(c)
    assert (
        repr(c)
        == f"""cirq.{circuit_cls.__name__}([
    cirq.Moment(
        cirq.Z(cirq.GridQubit(0, 0)),
    ),
], device={repr(FOXY)})"""
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_empty_moments(circuit_cls):
    # 1-qubit test
    op = cirq.X(cirq.NamedQubit('a'))
    op_moment = cirq.Moment([op])
    circuit = circuit_cls([op_moment, op_moment, cirq.Moment(), op_moment])

    cirq.testing.assert_has_diagram(circuit, "a: ───X───X───────X───", use_unicode_characters=True)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a
│
X
│
X
│
│
│
X
│
""",
        use_unicode_characters=True,
        transpose=True,
    )

    # 1-qubit ascii-only test
    cirq.testing.assert_has_diagram(circuit, "a: ---X---X-------X---", use_unicode_characters=False)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a
|
X
|
X
|
|
|
X
|
""",
        use_unicode_characters=False,
        transpose=True,
    )

    # 2-qubit test
    op = cirq.CNOT(cirq.NamedQubit('a'), cirq.NamedQubit('b'))
    op_moment = cirq.Moment([op])
    circuit = circuit_cls([op_moment, op_moment, cirq.Moment(), op_moment])

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───@───@───────@───
      │   │       │
b: ───X───X───────X───""",
        use_unicode_characters=True,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
a b
│ │
@─X
│ │
@─X
│ │
│ │
│ │
@─X
│ │
""",
        use_unicode_characters=True,
        transpose=True,
    )

    # 2-qubit ascii-only test
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ---@---@-------@---
      |   |       |
b: ---X---X-------X---""",
        use_unicode_characters=False,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
a b
| |
@-X
| |
@-X
| |
| |
| |
@-X
| |
""",
        use_unicode_characters=False,
        transpose=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_symbol_addition_in_gate_exponent(circuit_cls):
    # 1-qubit test
    qubit = cirq.NamedQubit('a')
    circuit = circuit_cls(
        cirq.X(qubit) ** 0.5,
        cirq.YPowGate(exponent=sympy.Symbol('a') + sympy.Symbol('b')).on(qubit),
    )
    cirq.testing.assert_has_diagram(
        circuit, 'a: ───X^0.5───Y^(a + b)───', use_unicode_characters=True
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
a
│
X^0.5
│
Y^(a + b)
│
""",
        use_unicode_characters=True,
        transpose=True,
    )

    cirq.testing.assert_has_diagram(
        circuit, 'a: ---X^0.5---Y^(a + b)---', use_unicode_characters=False
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
a
|
X^0.5
|
Y^(a + b)
|

 """,
        use_unicode_characters=False,
        transpose=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_slice(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(
        [
            cirq.Moment([cirq.H(a), cirq.H(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.H(b)]),
        ]
    )
    assert c[0:1] == circuit_cls([cirq.Moment([cirq.H(a), cirq.H(b)])])
    assert c[::2] == circuit_cls([cirq.Moment([cirq.H(a), cirq.H(b)]), cirq.Moment([cirq.H(b)])])
    assert c[0:1:2] == circuit_cls([cirq.Moment([cirq.H(a), cirq.H(b)])])
    assert c[1:3:] == circuit_cls([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.H(b)])])
    assert c[::-1] == circuit_cls(
        [
            cirq.Moment([cirq.H(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.H(a), cirq.H(b)]),
        ]
    )
    assert c[3:0:-1] == circuit_cls([cirq.Moment([cirq.H(b)]), cirq.Moment([cirq.CZ(a, b)])])
    assert c[0:2:-1] == circuit_cls()


def test_concatenate():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.Circuit()
    d = cirq.Circuit([cirq.Moment([cirq.X(b)])])
    e = cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)])])

    assert c + d == cirq.Circuit([cirq.Moment([cirq.X(b)])])
    assert d + c == cirq.Circuit([cirq.Moment([cirq.X(b)])])
    assert e + d == cirq.Circuit([cirq.Moment([cirq.X(a), cirq.X(b)]), cirq.Moment([cirq.X(b)])])

    d += c
    assert d == cirq.Circuit([cirq.Moment([cirq.X(b)])])

    c += d
    assert c == cirq.Circuit([cirq.Moment([cirq.X(b)])])

    f = e + d
    f += e
    assert f == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    with pytest.raises(TypeError):
        _ = c + 'a'


def test_concatenate_with_device():
    fox = cirq.Circuit(device=FOXY)
    cone = cirq.Circuit(device=BCONE)
    unr = cirq.Circuit()

    _ = cone + cone
    _ = cone + unr
    _ = unr + cone
    cone += unr
    with pytest.raises(ValueError):
        _ = cone + fox

    unr.append(cirq.X(cirq.NamedQubit('not_allowed')))
    with pytest.raises(ValueError):
        cone += unr
    with pytest.raises(ValueError):
        _ = cone + unr
    assert len(cone) == 0


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_with_device(circuit_cls):

    c = circuit_cls(cirq.X(cirq.LineQubit(0)))
    c2 = c.with_device(FOXY, lambda e: cirq.GridQubit(e.x, 0))
    assert c2 == circuit_cls(cirq.X(cirq.GridQubit(0, 0)), device=FOXY)

    # Qubit type must be correct.
    c = circuit_cls(cirq.X(cirq.LineQubit(0)))
    with pytest.raises(ValueError, match='Unsupported qubit type'):
        _ = c.with_device(FOXY)

    # Operations must be compatible from the start
    c = circuit_cls(cirq.X(cirq.GridQubit(0, 0)))
    _ = c.with_device(FOXY)
    c = circuit_cls(cirq.H(cirq.GridQubit(0, 0)))
    with pytest.raises(ValueError, match='Unsupported gate type'):
        _ = c.with_device(FOXY)

    # Some qubits exist on multiple devices.
    c = circuit_cls(cirq.X(cirq.GridQubit(0, 0)), device=FOXY)
    with pytest.raises(ValueError):
        _ = c.with_device(BCONE)
    c = circuit_cls(cirq.X(cirq.GridQubit(0, 6)), device=FOXY)
    _ = c.with_device(BCONE)


def test_set_device():
    c = cirq.Circuit(cirq.X(cirq.LineQubit(0)))
    assert c.device is cirq.UNCONSTRAINED_DEVICE

    with pytest.raises(ValueError):
        c.device = FOXY
    assert c.device is cirq.UNCONSTRAINED_DEVICE

    c[:] = []
    c.append(cirq.X(cirq.GridQubit(0, 0)))
    c.device = FOXY
    assert c.device == FOXY


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_multiply(circuit_cls):
    a = cirq.NamedQubit('a')

    c = circuit_cls()
    d = circuit_cls([cirq.Moment([cirq.X(a)])])

    assert c * 0 == circuit_cls()
    assert d * 0 == circuit_cls()
    assert d * 2 == circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)])])

    twice_copied_circuit = circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)])])
    for num in [np.int64(2), np.ushort(2), np.int8(2), np.int32(2), np.short(2)]:
        assert num * d == twice_copied_circuit
        assert d * num == twice_copied_circuit

    assert np.array([2])[0] * d == circuit_cls([cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)])])
    assert 1 * c == circuit_cls()
    assert -1 * d == circuit_cls()
    assert 1 * d == circuit_cls([cirq.Moment([cirq.X(a)])])

    d *= 3
    assert d == circuit_cls(
        [cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)]), cirq.Moment([cirq.X(a)])]
    )

    with pytest.raises(TypeError):
        _ = c * 'a'
    with pytest.raises(TypeError):
        _ = 'a' * c
    with pytest.raises(TypeError):
        c *= 'a'


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_container_methods(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(
        [
            cirq.Moment([cirq.H(a), cirq.H(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.H(b)]),
        ]
    )
    assert list(c) == list(c._moments)
    # __iter__
    assert list(iter(c)) == list(c._moments)
    # __reversed__ for free.
    assert list(reversed(c)) == list(reversed(c._moments))
    # __contains__ for free.
    assert cirq.Moment([cirq.H(b)]) in c

    assert len(c) == 3


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_bad_index(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls([cirq.Moment([cirq.H(a), cirq.H(b)])])
    with pytest.raises(TypeError):
        _ = c['string']


def test_append_strategies():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    stream = [cirq.X(a), cirq.CZ(a, b), cirq.X(b), cirq.X(b), cirq.X(a)]

    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment([cirq.X(a)]),
        ]
    )

    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment([cirq.X(b), cirq.X(a)]),
        ]
    )

    c = cirq.Circuit()
    c.append(stream, cirq.InsertStrategy.EARLIEST)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(b), cirq.X(a)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )


def test_insert_op_tree_new():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit()

    op_tree_list = [
        (-10, 0, cirq.CZ(a, b), a),
        (-20, 0, cirq.X(a), a),
        (20, 2, cirq.X(b), b),
        (2, 2, cirq.H(b), b),
        (-3, 1, cirq.H(a), a),
    ]

    for given_index, actual_index, operation, qubit in op_tree_list:
        c.insert(given_index, operation, cirq.InsertStrategy.NEW)
        assert c.operation_at(qubit, actual_index) == operation

    c.insert(1, (), cirq.InsertStrategy.NEW)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.H(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.H(b)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )

    BAD_INSERT = cirq.InsertStrategy('BAD', 'Bad strategy for testing.')
    with pytest.raises(ValueError):
        c.insert(1, cirq.X(a), BAD_INSERT)


def test_insert_op_tree_newinline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit()

    op_tree_list = [
        (-5, 0, [cirq.H(a), cirq.X(b)], [a, b]),
        (-15, 0, [cirq.CZ(a, b)], [a]),
        (15, 2, [cirq.H(b), cirq.X(a)], [b, a]),
    ]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.NEW_THEN_INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]

    c2 = cirq.Circuit()
    c2.insert(
        0,
        [cirq.CZ(a, b), cirq.H(a), cirq.X(b), cirq.H(b), cirq.X(a)],
        cirq.InsertStrategy.NEW_THEN_INLINE,
    )
    assert c == c2


def test_insert_op_tree_inline():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit([cirq.Moment([cirq.H(a)])])

    op_tree_list = [
        (1, 1, [cirq.H(a), cirq.X(b)], [a, b]),
        (0, 0, [cirq.X(b)], [b]),
        (4, 3, [cirq.H(b)], [b]),
        (5, 3, [cirq.H(a)], [a]),
        (-2, 0, [cirq.X(b)], [b]),
        (-5, 0, [cirq.CZ(a, b)], [a]),
    ]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.INLINE)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index) == op_list[i]


def test_insert_op_tree_earliest():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit([cirq.Moment([cirq.H(a)])])

    op_tree_list = [
        (5, [1, 0], [cirq.X(a), cirq.X(b)], [a, b]),
        (1, [1], [cirq.H(b)], [b]),
        (-4, [0], [cirq.X(b)], [b]),
    ]

    for given_index, actual_index, op_list, qubits in op_tree_list:
        c.insert(given_index, op_list, cirq.InsertStrategy.EARLIEST)
        for i in range(len(op_list)):
            assert c.operation_at(qubits[i], actual_index[i]) == op_list[i]


def test_insert_moment():
    a = cirq.NamedQubit('alice')
    b = cirq.NamedQubit('bob')
    c = cirq.Circuit()

    moment_list = [
        (-10, 0, [cirq.CZ(a, b)], a, cirq.InsertStrategy.NEW_THEN_INLINE),
        (-20, 0, [cirq.X(a)], a, cirq.InsertStrategy.NEW),
        (20, 2, [cirq.X(b)], b, cirq.InsertStrategy.INLINE),
        (2, 2, [cirq.H(b)], b, cirq.InsertStrategy.EARLIEST),
        (-3, 1, [cirq.H(a)], a, cirq.InsertStrategy.EARLIEST),
    ]

    for given_index, actual_index, operation, qubit, strat in moment_list:
        c.insert(given_index, cirq.Moment(operation), strat)
        assert c.operation_at(qubit, actual_index) == operation[0]


def test_insert_validates_all_operations_before_inserting():
    a, b = cirq.GridQubit(0, 0), cirq.GridQubit(1, 1)
    c = cirq.Circuit(device=FOXY)
    operations = [cirq.Z(a), cirq.CZ(a, b)]

    with pytest.raises(ValueError, match='Non-local interaction'):
        c.insert(0, operations)

    assert len(c) == 0


def test_insert_inline_near_start():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment(),
        ]
    )

    c.insert(1, cirq.X(a), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
        ]
    )

    c.insert(1, cirq.Y(a), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Y(a)]),
            cirq.Moment(),
        ]
    )

    c.insert(0, cirq.Z(b), strategy=cirq.InsertStrategy.INLINE)
    assert c == cirq.Circuit(
        [
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Y(a)]),
            cirq.Moment(),
        ]
    )


def test_insert_at_frontier_init():
    x = cirq.NamedQubit('x')
    op = cirq.X(x)
    circuit = cirq.Circuit(op)
    actual_frontier = circuit.insert_at_frontier(op, 3)
    expected_circuit = cirq.Circuit(
        [cirq.Moment([op]), cirq.Moment(), cirq.Moment(), cirq.Moment([op])]
    )
    assert circuit == expected_circuit
    expected_frontier = defaultdict(lambda: 0)
    expected_frontier[x] = 4
    assert actual_frontier == expected_frontier

    with pytest.raises(ValueError):
        circuit = cirq.Circuit([cirq.Moment(), cirq.Moment([op])])
        frontier = {x: 2}
        circuit.insert_at_frontier(op, 0, frontier)


def test_insert_at_frontier():
    class Replacer(cirq.PointOptimizer):
        def __init__(self, replacer=(lambda x: x)):
            super().__init__()
            self.replacer = replacer

        def optimization_at(
            self, circuit: 'cirq.Circuit', index: int, op: 'cirq.Operation'
        ) -> Optional['cirq.PointOptimizationSummary']:
            new_ops = self.replacer(op)
            return cirq.PointOptimizationSummary(
                clear_span=1, clear_qubits=op.qubits, new_operations=new_ops
            )

    replacer = lambda op: ((cirq.Z(op.qubits[0]),) * 2 + (op, cirq.Y(op.qubits[0])))
    prepend_two_Xs_append_one_Y = Replacer(replacer)
    qubits = [cirq.NamedQubit(s) for s in 'abcdef']
    a, b, c = qubits[:3]

    circuit = cirq.Circuit(
        [cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.CZ(b, c)]), cirq.Moment([cirq.CZ(a, b)])]
    )

    prepend_two_Xs_append_one_Y.optimize_circuit(circuit)

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───Z───Z───@───Y───────────────Z───Z───@───Y───
              │                           │
b: ───────────@───Z───Z───@───Y───────────@───────
                          │
c: ───────────────────────@───────────────────────
""",
    )

    prepender = lambda op: (cirq.X(op.qubits[0]),) * 3 + (op,)
    prepend_3_Xs = Replacer(prepender)
    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.CNOT(a, b)]),
            cirq.Moment([cirq.CNOT(b, c)]),
            cirq.Moment([cirq.CNOT(c, b)]),
        ]
    )
    prepend_3_Xs.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───X───X───X───@───────────────────────────────────
                  │
b: ───────────────X───X───X───X───@───────────────X───
                                  │               │
c: ───────────────────────────────X───X───X───X───@───
""",
    )

    duplicate = Replacer(lambda op: (op,) * 2)
    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.CZ(qubits[j], qubits[j + 1]) for j in range(i % 2, 5, 2)])
            for i in range(4)
        ]
    )

    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───@───@───────────@───@───────────
      │   │           │   │
b: ───@───@───@───@───@───@───@───@───
              │   │           │   │
c: ───@───@───@───@───@───@───@───@───
      │   │           │   │
d: ───@───@───@───@───@───@───@───@───
              │   │           │   │
e: ───@───@───@───@───@───@───@───@───
      │   │           │   │
f: ───@───@───────────@───@───────────
""",
    )

    circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.CZ(*qubits[2:4]), cirq.CNOT(*qubits[:2])]),
            cirq.Moment([cirq.CNOT(*qubits[1::-1])]),
        ]
    )

    duplicate.optimize_circuit(circuit)
    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───@───@───X───X───
      │   │   │   │
b: ───X───X───@───@───

c: ───@───────@───────
      │       │
d: ───@───────@───────
""",
    )


def test_insert_into_range():
    x = cirq.NamedQubit('x')
    y = cirq.NamedQubit('y')
    c = cirq.Circuit([cirq.Moment([cirq.X(x)])] * 4)
    c.insert_into_range([cirq.Z(x), cirq.CZ(x, y)], 2, 2)
    cirq.testing.assert_has_diagram(
        c,
        """
x: ───X───X───Z───@───X───X───
                  │
y: ───────────────@───────────
""",
    )

    c.insert_into_range([cirq.Y(y), cirq.Y(y), cirq.Y(y), cirq.CX(y, x)], 1, 4)
    cirq.testing.assert_has_diagram(
        c,
        """
x: ───X───X───Z───@───X───X───X───
                  │       │
y: ───────Y───Y───@───Y───@───────
""",
    )

    c.insert_into_range([cirq.H(y), cirq.H(y)], 6, 7)
    cirq.testing.assert_has_diagram(
        c,
        """
x: ───X───X───Z───@───X───X───X───────
                  │       │
y: ───────Y───Y───@───Y───@───H───H───
""",
    )

    c.insert_into_range([cirq.T(y)], 0, 1)
    cirq.testing.assert_has_diagram(
        c,
        """
x: ───X───X───Z───@───X───X───X───────
                  │       │
y: ───T───Y───Y───@───Y───@───H───H───
""",
    )

    with pytest.raises(IndexError):
        c.insert_into_range([cirq.CZ(x, y)], 10, 10)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_next_moment_operating_on(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls()
    assert c.next_moment_operating_on([a]) is None
    assert c.next_moment_operating_on([a], 0) is None
    assert c.next_moment_operating_on([a], 102) is None

    c = circuit_cls([cirq.Moment([cirq.X(a)])])
    assert c.next_moment_operating_on([a]) == 0
    assert c.next_moment_operating_on([a], 0) == 0
    assert c.next_moment_operating_on([a, b]) == 0
    assert c.next_moment_operating_on([a], 1) is None
    assert c.next_moment_operating_on([b]) is None

    c = circuit_cls(
        [cirq.Moment(), cirq.Moment([cirq.X(a)]), cirq.Moment(), cirq.Moment([cirq.CZ(a, b)])]
    )

    assert c.next_moment_operating_on([a], 0) == 1
    assert c.next_moment_operating_on([a], 1) == 1
    assert c.next_moment_operating_on([a], 2) == 3
    assert c.next_moment_operating_on([a], 3) == 3
    assert c.next_moment_operating_on([a], 4) is None

    assert c.next_moment_operating_on([b], 0) == 3
    assert c.next_moment_operating_on([b], 1) == 3
    assert c.next_moment_operating_on([b], 2) == 3
    assert c.next_moment_operating_on([b], 3) == 3
    assert c.next_moment_operating_on([b], 4) is None

    assert c.next_moment_operating_on([a, b], 0) == 1
    assert c.next_moment_operating_on([a, b], 1) == 1
    assert c.next_moment_operating_on([a, b], 2) == 3
    assert c.next_moment_operating_on([a, b], 3) == 3
    assert c.next_moment_operating_on([a, b], 4) is None


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_next_moment_operating_on_distance(circuit_cls):
    a = cirq.NamedQubit('a')

    c = circuit_cls(
        [
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
        ]
    )

    assert c.next_moment_operating_on([a], 0, max_distance=4) is None
    assert c.next_moment_operating_on([a], 1, max_distance=3) is None
    assert c.next_moment_operating_on([a], 2, max_distance=2) is None
    assert c.next_moment_operating_on([a], 3, max_distance=1) is None
    assert c.next_moment_operating_on([a], 4, max_distance=0) is None

    assert c.next_moment_operating_on([a], 0, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=4) == 4
    assert c.next_moment_operating_on([a], 2, max_distance=3) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=2) == 4
    assert c.next_moment_operating_on([a], 4, max_distance=1) == 4

    assert c.next_moment_operating_on([a], 5, max_distance=0) is None
    assert c.next_moment_operating_on([a], 1, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 3, max_distance=5) == 4
    assert c.next_moment_operating_on([a], 1, max_distance=500) == 4

    # Huge max distances should be handled quickly due to capping.
    assert c.next_moment_operating_on([a], 5, max_distance=10 ** 100) is None

    with pytest.raises(ValueError, match='Negative max_distance'):
        c.next_moment_operating_on([a], 0, max_distance=-1)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_prev_moment_operating_on(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls()
    assert c.prev_moment_operating_on([a]) is None
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([a], 102) is None

    c = circuit_cls([cirq.Moment([cirq.X(a)])])
    assert c.prev_moment_operating_on([a]) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a, b]) == 0
    assert c.prev_moment_operating_on([a], 0) is None
    assert c.prev_moment_operating_on([b]) is None

    c = circuit_cls(
        [
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment(),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
        ]
    )

    assert c.prev_moment_operating_on([a], 4) == 2
    assert c.prev_moment_operating_on([a], 3) == 2
    assert c.prev_moment_operating_on([a], 2) == 0
    assert c.prev_moment_operating_on([a], 1) == 0
    assert c.prev_moment_operating_on([a], 0) is None

    assert c.prev_moment_operating_on([b], 4) == 0
    assert c.prev_moment_operating_on([b], 3) == 0
    assert c.prev_moment_operating_on([b], 2) == 0
    assert c.prev_moment_operating_on([b], 1) == 0
    assert c.prev_moment_operating_on([b], 0) is None

    assert c.prev_moment_operating_on([a, b], 4) == 2
    assert c.prev_moment_operating_on([a, b], 3) == 2
    assert c.prev_moment_operating_on([a, b], 2) == 0
    assert c.prev_moment_operating_on([a, b], 1) == 0
    assert c.prev_moment_operating_on([a, b], 0) is None

    with pytest.raises(ValueError, match='Negative max_distance'):
        assert c.prev_moment_operating_on([a, b], 4, max_distance=-1)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_prev_moment_operating_on_distance(circuit_cls):
    a = cirq.NamedQubit('a')

    c = circuit_cls(
        [
            cirq.Moment(),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
        ]
    )

    assert c.prev_moment_operating_on([a], max_distance=4) is None
    assert c.prev_moment_operating_on([a], 6, max_distance=4) is None
    assert c.prev_moment_operating_on([a], 5, max_distance=3) is None
    assert c.prev_moment_operating_on([a], 4, max_distance=2) is None
    assert c.prev_moment_operating_on([a], 3, max_distance=1) is None
    assert c.prev_moment_operating_on([a], 2, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 1, max_distance=0) is None
    assert c.prev_moment_operating_on([a], 0, max_distance=0) is None

    assert c.prev_moment_operating_on([a], 6, max_distance=5) == 1
    assert c.prev_moment_operating_on([a], 5, max_distance=4) == 1
    assert c.prev_moment_operating_on([a], 4, max_distance=3) == 1
    assert c.prev_moment_operating_on([a], 3, max_distance=2) == 1
    assert c.prev_moment_operating_on([a], 2, max_distance=1) == 1

    assert c.prev_moment_operating_on([a], 6, max_distance=10) == 1
    assert c.prev_moment_operating_on([a], 6, max_distance=100) == 1
    assert c.prev_moment_operating_on([a], 13, max_distance=500) == 1

    # Huge max distances should be handled quickly due to capping.
    assert c.prev_moment_operating_on([a], 1, max_distance=10 ** 100) is None

    with pytest.raises(ValueError, match='Negative max_distance'):
        c.prev_moment_operating_on([a], 6, max_distance=-1)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_operation_at(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls()
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, -1) is None
    assert c.operation_at(a, 102) is None

    c = circuit_cls([cirq.Moment()])
    assert c.operation_at(a, 0) is None

    c = circuit_cls([cirq.Moment([cirq.X(a)])])
    assert c.operation_at(b, 0) is None
    assert c.operation_at(a, 1) is None
    assert c.operation_at(a, 0) == cirq.X(a)

    c = circuit_cls([cirq.Moment(), cirq.Moment([cirq.CZ(a, b)])])
    assert c.operation_at(a, 0) is None
    assert c.operation_at(a, 1) == cirq.CZ(a, b)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_findall_operations(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)
    za = cirq.Z.on(a)
    zb = cirq.Z.on(b)

    def is_x(op: cirq.Operation) -> bool:
        return isinstance(op, cirq.GateOperation) and isinstance(op.gate, cirq.XPowGate)

    c = circuit_cls()
    assert list(c.findall_operations(is_x)) == []

    c = circuit_cls(xa)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = circuit_cls(za)
    assert list(c.findall_operations(is_x)) == []

    c = circuit_cls([za, zb] * 8)
    assert list(c.findall_operations(is_x)) == []

    c = circuit_cls(xa, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (0, xb)]

    c = circuit_cls(xa, zb)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = circuit_cls(xa, za)
    assert list(c.findall_operations(is_x)) == [(0, xa)]

    c = circuit_cls([xa] * 8)
    assert list(c.findall_operations(is_x)) == list(enumerate([xa] * 8))

    c = circuit_cls(za, zb, xa, xb)
    assert list(c.findall_operations(is_x)) == [(1, xa), (1, xb)]

    c = circuit_cls(xa, zb, za, xb)
    assert list(c.findall_operations(is_x)) == [(0, xa), (1, xb)]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_findall_operations_with_gate(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(a), cirq.Z(b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.measure(a), cirq.measure(b)]),
        ]
    )
    assert list(c.findall_operations_with_gate_type(cirq.XPowGate)) == [
        (0, cirq.X(a), cirq.X),
        (2, cirq.X(a), cirq.X),
        (2, cirq.X(b), cirq.X),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.CZPowGate)) == [
        (3, cirq.CZ(a, b), cirq.CZ),
    ]
    assert list(c.findall_operations_with_gate_type(cirq.MeasurementGate)) == [
        (4, cirq.MeasurementGate(1, key='a').on(a), cirq.MeasurementGate(1, key='a')),
        (4, cirq.MeasurementGate(1, key='b').on(b), cirq.MeasurementGate(1, key='b')),
    ]


def assert_findall_operations_until_blocked_as_expected(
    circuit=None, start_frontier=None, is_blocker=None, expected_ops=None
):
    if circuit is None:
        circuit = cirq.Circuit()
    if start_frontier is None:
        start_frontier = {}
    kwargs = {} if is_blocker is None else {'is_blocker': is_blocker}
    found_ops = circuit.findall_operations_until_blocked(start_frontier, **kwargs)

    for i, op in found_ops:
        assert i >= min((start_frontier[q] for q in op.qubits if q in start_frontier), default=0)
        assert set(op.qubits).intersection(start_frontier)

    if expected_ops is None:
        return
    assert sorted(found_ops) == sorted(expected_ops)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_findall_operations_until_blocked(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)

    assert_findall_operations_until_blocked_as_expected()

    circuit = circuit_cls(
        cirq.H(a),
        cirq.CZ(a, b),
        cirq.H(b),
        cirq.CZ(b, c),
        cirq.H(c),
        cirq.CZ(c, d),
        cirq.H(d),
        cirq.CZ(c, d),
        cirq.H(c),
        cirq.CZ(b, c),
        cirq.H(b),
        cirq.CZ(a, b),
        cirq.H(a),
    )
    expected_diagram = """
0: ───H───@───────────────────────────────────────@───H───
          │                                       │
1: ───────@───H───@───────────────────────@───H───@───────
                  │                       │
2: ───────────────@───H───@───────@───H───@───────────────
                          │       │
3: ───────────────────────@───H───@───────────────────────
""".strip()
    #     0   1   2   3   4   5   6   7   8   9   10  11  12
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    # Always return true to test basic features
    go_to_end = lambda op: False
    stop_if_op = lambda op: True
    stop_if_h_on_a = lambda op: op.gate == cirq.H and a in op.qubits

    # Empty cases.
    assert_findall_operations_until_blocked_as_expected(is_blocker=go_to_end, expected_ops=[])
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit, is_blocker=go_to_end, expected_ops=[]
    )

    # Clamped input cases. (out of bounds)
    assert_findall_operations_until_blocked_as_expected(
        start_frontier={a: 5}, is_blocker=stop_if_op, expected_ops=[]
    )
    assert_findall_operations_until_blocked_as_expected(
        start_frontier={a: -100}, is_blocker=stop_if_op, expected_ops=[]
    )
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit, start_frontier={a: 100}, is_blocker=stop_if_op, expected_ops=[]
    )

    # Test if all operations are blocked
    for idx in range(0, 15):
        for q in (a, b, c, d):
            assert_findall_operations_until_blocked_as_expected(
                circuit=circuit, start_frontier={q: idx}, is_blocker=stop_if_op, expected_ops=[]
            )
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={a: idx, b: idx, c: idx, d: idx},
            is_blocker=stop_if_op,
            expected_ops=[],
        )

    # Cases where nothing is blocked, it goes to the end
    a_ending_ops = [(11, cirq.CZ.on(a, b)), (12, cirq.H.on(a))]
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={a: idx},
            is_blocker=go_to_end,
            expected_ops=a_ending_ops,
        )

    # Block on H, but pick up the CZ
    for idx in range(2, 10):
        assert_findall_operations_until_blocked_as_expected(
            circuit=circuit,
            start_frontier={a: idx},
            is_blocker=stop_if_h_on_a,
            expected_ops=[(11, cirq.CZ.on(a, b))],
        )

    circuit = circuit_cls([cirq.CZ(a, b), cirq.CZ(a, b), cirq.CZ(b, c)])
    expected_diagram = """
0: ───@───@───────
      │   │
1: ───@───@───@───
              │
2: ───────────@───
""".strip()
    #     0   1   2
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0}
    is_blocker = lambda next_op: sorted(next_op.qubits) != [a, b]
    expected_ops = [(0, cirq.CZ(a, b)), (1, cirq.CZ(a, b))]
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=expected_ops,
    )

    circuit = circuit_cls([cirq.ZZ(a, b), cirq.ZZ(b, c)])
    expected_diagram = """
0: ───ZZ────────
      │
1: ───ZZ───ZZ───
           │
2: ────────ZZ───
""".strip()
    #     0    1
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0, c: 0}
    is_blocker = lambda op: a in op.qubits
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=[]
    )

    circuit = circuit_cls([cirq.ZZ(a, b), cirq.XX(c, d), cirq.ZZ(b, c), cirq.Z(b)])
    expected_diagram = """
0: ───ZZ────────────
      │
1: ───ZZ───ZZ───Z───
           │
2: ───XX───ZZ───────
      │
3: ───XX────────────
""".strip()
    #     0    1    2
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, b: 0, c: 0, d: 0}
    is_blocker = lambda op: isinstance(op.gate, cirq.XXPowGate)
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit,
        start_frontier=start_frontier,
        is_blocker=is_blocker,
        expected_ops=[(0, cirq.ZZ(a, b))],
    )

    circuit = circuit_cls([cirq.XX(a, b), cirq.Z(a), cirq.ZZ(b, c), cirq.ZZ(c, d), cirq.Z(d)])
    expected_diagram = """
0: ───XX───Z─────────────
      │
1: ───XX───ZZ────────────
           │
2: ────────ZZ───ZZ───────
                │
3: ─────────────ZZ───Z───
""".strip()
    #     0    1    2    3
    cirq.testing.assert_has_diagram(circuit, expected_diagram)

    start_frontier = {a: 0, d: 0}
    assert_findall_operations_until_blocked_as_expected(
        circuit=circuit, start_frontier=start_frontier, is_blocker=is_blocker, expected_ops=[]
    )


@pytest.mark.parametrize('seed', [randint(0, 2 ** 31)])
@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_findall_operations_until_blocked_docstring_examples(seed, circuit_cls):
    prng = np.random.RandomState(seed)

    class ExampleGate(cirq.Gate):
        def __init__(self, n_qubits, label):
            self.n_qubits = n_qubits
            self.label = label

        def num_qubits(self):
            return self.n_qubits

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(wire_symbols=[self.label] * self.n_qubits)

    def is_blocker(op):
        if op.gate.label == 'F':
            return False
        if op.gate.label == 'T':
            return True
        return prng.rand() < 0.5

    F2 = ExampleGate(2, 'F')
    T2 = ExampleGate(2, 'T')
    M2 = ExampleGate(2, 'M')
    a, b, c, d = cirq.LineQubit.range(4)

    circuit = circuit_cls([F2(a, b), F2(a, b), T2(b, c)])
    start = {a: 0, b: 0}
    expected_diagram = """
0: ───F───F───────
      │   │
1: ───F───F───T───
              │
2: ───────────T───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    expected_ops = [(0, F2(a, b)), (1, F2(a, b))]
    new_circuit = circuit_cls([op for _, op in expected_ops])
    expected_diagram = """
0: ───F───F───
      │   │
1: ───F───F───
    """
    cirq.testing.assert_has_diagram(new_circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == expected_ops

    circuit = circuit_cls([M2(a, b), M2(b, c), F2(a, b), M2(c, d)])
    start = {a: 2, b: 2}
    expected_diagram = """
0: ───M───────F───
      │       │
1: ───M───M───F───
          │
2: ───────M───M───
              │
3: ───────────M───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    expected_ops = [(2, F2(a, b))]
    new_circuit = circuit_cls([op for _, op in expected_ops])
    expected_diagram = """
0: ───F───
      │
1: ───F───
    """
    cirq.testing.assert_has_diagram(new_circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == expected_ops

    circuit = circuit_cls([M2(a, b), T2(b, c), M2(a, b), M2(c, d)])
    start = {a: 1, b: 1}
    expected_diagram = """
0: ───M───────M───
      │       │
1: ───M───T───M───
          │
2: ───────T───M───
              │
3: ───────────M───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == []

    ops = [(0, F2(a, b)), (1, F2(a, b))]
    circuit = circuit_cls([op for _, op in ops])
    start = {a: 0, b: 1}
    expected_diagram = """
0: ───F───F───
      │   │
1: ───F───F───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == ops

    ops = [F2(a, b), F2(b, c), F2(c, d)]
    circuit = circuit_cls(ops)
    start = {a: 0, d: 0}
    expected_diagram = """
0: ───F───────────
      │
1: ───F───F───────
          │
2: ───────F───F───
              │
3: ───────────F───
    """
    cirq.testing.assert_has_diagram(circuit, expected_diagram)
    assert circuit.findall_operations_until_blocked(start, is_blocker) == [
        (0, F2(a, b)),
        (2, F2(c, d)),
    ]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_has_measurements(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ma = cirq.measure(a)
    mb = cirq.measure(b)

    c = circuit_cls()
    assert not c.has_measurements()

    c = circuit_cls(xa, xb)
    assert not c.has_measurements()

    c = circuit_cls(ma)
    assert c.has_measurements()

    c = circuit_cls(ma, mb)
    assert c.has_measurements()

    c = circuit_cls(xa, ma)
    assert c.has_measurements()

    c = circuit_cls(xa, ma, xb, mb)
    assert c.has_measurements()

    c = circuit_cls(ma, xa)
    assert c.has_measurements()

    c = circuit_cls(ma, xa, mb)
    assert c.has_measurements()

    c = circuit_cls(xa, ma, xb, xa)
    assert c.has_measurements()

    c = circuit_cls(ma, ma)
    assert c.has_measurements()

    c = circuit_cls(xa, ma, xa)
    assert c.has_measurements()


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_are_all_or_any_measurements_terminal(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ma = cirq.measure(a)
    mb = cirq.measure(b)

    c = circuit_cls()
    assert c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()

    c = circuit_cls(xa, xb)
    assert c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()

    c = circuit_cls(ma)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(ma, mb)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(xa, ma)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(xa, ma, xb, mb)
    assert c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(ma, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()

    c = circuit_cls(ma, xa, mb)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(xa, ma, xb, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()

    c = circuit_cls(ma, ma)
    assert not c.are_all_measurements_terminal()
    assert c.are_any_measurements_terminal()

    c = circuit_cls(xa, ma, xa)
    assert not c.are_all_measurements_terminal()
    assert not c.are_any_measurements_terminal()


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_all_or_any_terminal(circuit_cls):
    def is_x_pow_gate(op):
        return isinstance(op.gate, cirq.XPowGate)

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    xa = cirq.X.on(a)
    xb = cirq.X.on(b)

    ya = cirq.Y.on(a)
    yb = cirq.Y.on(b)

    c = circuit_cls()
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(ya)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(ya, yb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(ya, yb, xa)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(ya, yb, xa, xb)
    assert c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xa, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xa, ya)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xb, ya, yb)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert not c.are_any_matches_terminal(is_x_pow_gate)

    c = circuit_cls(xa, ya, xa)
    assert not c.are_all_matches_terminal(is_x_pow_gate)
    assert c.are_any_matches_terminal(is_x_pow_gate)

    def is_circuit_op(op):
        isinstance(op, cirq.CircuitOperation)

    cop_1 = cirq.CircuitOperation(cirq.FrozenCircuit(xa, ya))
    cop_2 = cirq.CircuitOperation(cirq.FrozenCircuit(cop_1, xb))
    c = circuit_cls(cop_2, yb)
    # are_all_matches_terminal treats CircuitOperations as transparent.
    assert c.are_all_matches_terminal(is_circuit_op)
    assert not c.are_any_matches_terminal(is_circuit_op)


def test_clear_operations_touching():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = cirq.Circuit()
    c.clear_operations_touching([a, b], range(10))
    assert c == cirq.Circuit()

    c = cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment(),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment(),
        ]
    )
    c.clear_operations_touching([a], [1, 3, 4, 6, 7])
    assert c == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment(),
        ]
    )

    c = cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment(),
            cirq.Moment([cirq.X(b)]),
            cirq.Moment(),
        ]
    )
    c.clear_operations_touching([a, b], [1, 3, 4, 6, 7])
    assert c == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.X(a)]),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
        ]
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_all_qubits(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )
    assert c.all_qubits() == {a, b}

    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(a)]),
        ]
    )
    assert c.all_qubits() == {a}

    c = circuit_cls(
        [
            cirq.Moment([cirq.CZ(a, b)]),
        ]
    )
    assert c.all_qubits() == {a, b}

    c = circuit_cls([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a)])])
    assert c.all_qubits() == {a, b}


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_all_operations(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(b)]

    c = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(a)]),
        ]
    )
    assert list(c.all_operations()) == [cirq.X(a), cirq.X(a)]

    c = circuit_cls(
        [
            cirq.Moment([cirq.CZ(a, b)]),
        ]
    )
    assert list(c.all_operations()) == [cirq.CZ(a, b)]

    c = circuit_cls([cirq.Moment([cirq.CZ(a, b)]), cirq.Moment([cirq.X(a)])])
    assert list(c.all_operations()) == [cirq.CZ(a, b), cirq.X(a)]

    c = circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CNOT(a, b)]),
            cirq.Moment([cirq.Z(b), cirq.H(a)]),  # Different qubit order
            cirq.Moment([]),
        ]
    )

    assert list(c.all_operations()) == [cirq.X(a), cirq.Y(b), cirq.CNOT(a, b), cirq.Z(b), cirq.H(a)]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_qid_shape_qubit(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    circuit = circuit_cls(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )

    assert cirq.qid_shape(circuit) == (2, 2)
    assert cirq.num_qubits(circuit) == 2
    assert circuit.qid_shape() == (2, 2)
    assert circuit.qid_shape(qubit_order=[c, a, b]) == (2, 2, 2)
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[a])


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_qid_shape_qudit(circuit_cls):
    class PlusOneMod3Gate(cirq.SingleQubitGate):
        def _qid_shape_(self):
            return (3,)

    class C2NotGate(cirq.Gate):
        def _qid_shape_(self):
            return (3, 2)

    class IdentityGate(cirq.SingleQubitGate):
        def _qid_shape_(self):
            return (1,)

    a, b, c = cirq.LineQid.for_qid_shape((3, 2, 1))

    circuit = circuit_cls(
        PlusOneMod3Gate().on(a),
        C2NotGate().on(a, b),
        IdentityGate().on_each(c),
    )

    assert cirq.num_qubits(circuit) == 3
    assert cirq.qid_shape(circuit) == (3, 2, 1)
    assert circuit.qid_shape() == (3, 2, 1)
    assert circuit.qid_shape()
    with pytest.raises(ValueError, match='extra qubits'):
        _ = circuit.qid_shape(qubit_order=[b, c])


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_teleportation_to_diagram(circuit_cls):
    ali = cirq.NamedQubit('(0, 0)')
    bob = cirq.NamedQubit('(0, 1)')
    msg = cirq.NamedQubit('(1, 0)')
    tmp = cirq.NamedQubit('(1, 1)')

    c = circuit_cls(
        [
            cirq.Moment([cirq.H(ali)]),
            cirq.Moment([cirq.CNOT(ali, bob)]),
            cirq.Moment([cirq.X(msg) ** 0.5]),
            cirq.Moment([cirq.CNOT(msg, ali)]),
            cirq.Moment([cirq.H(msg)]),
            cirq.Moment([cirq.measure(msg), cirq.measure(ali)]),
            cirq.Moment([cirq.CNOT(ali, bob)]),
            cirq.Moment([cirq.CNOT(msg, tmp)]),
            cirq.Moment([cirq.CZ(bob, tmp)]),
        ]
    )

    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0): ───H───@───────────X───────M───@───────────
               │           │           │
(0, 1): ───────X───────────┼───────────X───────@───
                           │                   │
(1, 0): ───────────X^0.5───@───H───M───────@───┼───
                                           │   │
(1, 1): ───────────────────────────────────X───@───
""",
    )

    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0): ---H---@-----------X-------M---@-----------
               |           |           |
(0, 1): -------X-----------|-----------X-------@---
                           |                   |
(1, 0): -----------X^0.5---@---H---M-------@---|---
                                           |   |
(1, 1): -----------------------------------X---@---
""",
        use_unicode_characters=False,
    )

    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0) (0, 1) (1, 0) (1, 1)
|      |      |      |
H      |      |      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      X^0.5  |
|      |      |      |
X-------------@      |
|      |      |      |
|      |      H      |
|      |      |      |
M      |      M      |
|      |      |      |
@------X      |      |
|      |      |      |
|      |      @------X
|      |      |      |
|      @-------------@
|      |      |      |
""",
        use_unicode_characters=False,
        transpose=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_with_unknown_exponent(circuit_cls):
    class WeirdGate(cirq.SingleQubitGate):
        def _circuit_diagram_info_(
            self, args: cirq.CircuitDiagramInfoArgs
        ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('B',), exponent='fancy')

    class WeirderGate(cirq.SingleQubitGate):
        def _circuit_diagram_info_(
            self, args: cirq.CircuitDiagramInfoArgs
        ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=('W',), exponent='fancy-that')

    c = circuit_cls(
        WeirdGate().on(cirq.NamedQubit('q')),
        WeirderGate().on(cirq.NamedQubit('q')),
    )

    # The hyphen in the exponent should cause parens to appear.
    cirq.testing.assert_has_diagram(c, 'q: ───B^fancy───W^(fancy-that)───')


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_circuit_diagram_on_gate_without_info(circuit_cls):
    q = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')

    class FGate(cirq.Gate):
        def __init__(self, num_qubits=1):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def __repr__(self):
            return 'python-object-FGate:arbitrary-digits'

    # Fallback to repr.
    f = FGate()
    cirq.testing.assert_has_diagram(
        circuit_cls(
            [
                cirq.Moment([f.on(q)]),
            ]
        ),
        """
(0, 0): ---python-object-FGate:arbitrary-digits---
""",
        use_unicode_characters=False,
    )

    f3 = FGate(3)
    # When used on multiple qubits, show the qubit order as a digit suffix.
    cirq.testing.assert_has_diagram(
        circuit_cls(
            [
                cirq.Moment([f3.on(q, q3, q2)]),
            ]
        ),
        """
(0, 0): ---python-object-FGate:arbitrary-digits---
           |
(0, 1): ---#3-------------------------------------
           |
(0, 2): ---#2-------------------------------------
""",
        use_unicode_characters=False,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_multi_qubit_gate(circuit_cls):
    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = circuit_cls(cirq.measure(q1, q2, q3, key='msg'))
    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0): ───M('msg')───
           │
(0, 1): ───M──────────
           │
(0, 2): ───M──────────
""",
    )
    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0): ---M('msg')---
           |
(0, 1): ---M----------
           |
(0, 2): ---M----------
""",
        use_unicode_characters=False,
    )
    cirq.testing.assert_has_diagram(
        c,
        """
(0, 0)   (0, 1) (0, 2)
│        │      │
M('msg')─M──────M
│        │      │
""",
        transpose=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_many_qubits_gate_but_multiple_wire_symbols(circuit_cls):
    class BadGate(cirq.testing.ThreeQubitGate):
        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, str]:
            return 'a', 'a'

    q1 = cirq.NamedQubit('(0, 0)')
    q2 = cirq.NamedQubit('(0, 1)')
    q3 = cirq.NamedQubit('(0, 2)')
    c = circuit_cls([cirq.Moment([BadGate().on(q1, q2, q3)])])
    with pytest.raises(ValueError, match='BadGate'):
        c.to_text_diagram()


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_parameterized_value(circuit_cls):
    q = cirq.NamedQubit('cube')

    class PGate(cirq.SingleQubitGate):
        def __init__(self, val):
            self.val = val

        def _circuit_diagram_info_(
            self, args: cirq.CircuitDiagramInfoArgs
        ) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(('P',), self.val)

    c = circuit_cls(
        PGate(1).on(q),
        PGate(2).on(q),
        PGate(sympy.Symbol('a')).on(q),
        PGate(sympy.Symbol('%$&#*(')).on(q),
    )
    assert str(c).strip() == 'cube: ───P───P^2───P^a───P^(%$&#*()───'


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_text_diagram_custom_order(circuit_cls):
    qa = cirq.NamedQubit('2')
    qb = cirq.NamedQubit('3')
    qc = cirq.NamedQubit('4')

    c = circuit_cls([cirq.Moment([cirq.X(qa), cirq.X(qb), cirq.X(qc)])])
    cirq.testing.assert_has_diagram(
        c,
        """
3: ---X---

4: ---X---

2: ---X---
""",
        qubit_order=cirq.QubitOrder.sorted_by(lambda e: int(str(e)) % 3),
        use_unicode_characters=False,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_overly_precise_diagram(circuit_cls):
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = circuit_cls([cirq.Moment([cirq.X(qa) ** 0.12345678])])
    cirq.testing.assert_has_diagram(
        c,
        """
a: ---X^0.123---
""",
        use_unicode_characters=False,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_none_precision_diagram(circuit_cls):
    # Test default precision of 3
    qa = cirq.NamedQubit('a')
    c = circuit_cls([cirq.Moment([cirq.X(qa) ** 0.4921875])])
    cirq.testing.assert_has_diagram(
        c,
        """
a: ---X^0.4921875---
""",
        use_unicode_characters=False,
        precision=None,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_custom_precision(circuit_cls):
    qa = cirq.NamedQubit('a')
    c = circuit_cls([cirq.Moment([cirq.X(qa) ** 0.12341234])])
    cirq.testing.assert_has_diagram(
        c,
        """
a: ---X^0.12341---
""",
        use_unicode_characters=False,
        precision=5,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_wgate(circuit_cls):
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(exponent=0.12341234, phase_exponent=0.43214321)
    c = circuit_cls([cirq.Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(
        c,
        """
a: ---PhX(0.43)^(1/8)---
""",
        use_unicode_characters=False,
        precision=2,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_wgate_none_precision(circuit_cls):
    qa = cirq.NamedQubit('a')
    test_wgate = cirq.PhasedXPowGate(exponent=0.12341234, phase_exponent=0.43214321)
    c = circuit_cls([cirq.Moment([test_wgate.on(qa)])])
    cirq.testing.assert_has_diagram(
        c,
        """
a: ---PhX(0.43214321)^0.12341234---
""",
        use_unicode_characters=False,
        precision=None,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_diagram_global_phase(circuit_cls):
    qa = cirq.NamedQubit('a')
    global_phase = cirq.GlobalPhaseOperation(coefficient=1j)
    c = circuit_cls([global_phase])
    cirq.testing.assert_has_diagram(
        c, "\n\nglobal phase:   0.5pi", use_unicode_characters=False, precision=2
    )
    cirq.testing.assert_has_diagram(
        c, "\n\nglobal phase:   0.5π", use_unicode_characters=True, precision=2
    )

    c = circuit_cls([cirq.X(qa), global_phase, global_phase])
    cirq.testing.assert_has_diagram(
        c,
        """\
a: ─────────────X───

global phase:   π""",
        use_unicode_characters=True,
        precision=2,
    )
    c = circuit_cls([cirq.X(qa), global_phase], cirq.Moment([cirq.X(qa), global_phase]))
    cirq.testing.assert_has_diagram(
        c,
        """\
a: ─────────────X──────X──────

global phase:   0.5π   0.5π
""",
        use_unicode_characters=True,
        precision=2,
    )

    c = circuit_cls(
        cirq.X(cirq.LineQubit(2)),
        cirq.CircuitOperation(circuit_cls(cirq.GlobalPhaseOperation(-1).with_tags("tag")).freeze()),
    )
    cirq.testing.assert_has_diagram(
        c,
        """\
2: ───X──────────

      π['tag']""",
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_has_unitary(circuit_cls):
    class NonUnitary(cirq.SingleQubitGate):
        pass

    class EventualUnitary(cirq.SingleQubitGate):
        def _decompose_(self, qubits):
            return cirq.X.on_each(*qubits)

    q = cirq.NamedQubit('q')

    # Non-unitary operations cause a non-unitary circuit.
    assert cirq.has_unitary(circuit_cls(cirq.X(q)))
    assert not cirq.has_unitary(circuit_cls(NonUnitary().on(q)))

    # Terminal measurements are ignored, though.
    assert cirq.has_unitary(circuit_cls(cirq.measure(q)))
    assert not cirq.has_unitary(circuit_cls(cirq.measure(q), cirq.measure(q)))

    # Still unitary if operations decompose into unitary operations.
    assert cirq.has_unitary(circuit_cls(EventualUnitary().on(q)))


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_text_diagram_jupyter(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    circuit = circuit_cls((cirq.CNOT(a, b), cirq.CNOT(b, c), cirq.CNOT(c, a)) * 50)
    text_expected = circuit.to_text_diagram()

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    circuit._repr_pretty_(p, False)
    assert p.text_pretty == text_expected

    # Test cycle handling
    p = FakePrinter()
    circuit._repr_pretty_(p, True)
    assert p.text_pretty == f'{circuit_cls.__name__}(...)'

    # Test Jupyter notebook html output
    text_html = circuit._repr_html_()
    # Don't enforce specific html surrounding the diagram content
    assert text_expected in text_html


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_circuit_to_unitary_matrix(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Single qubit gates.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a) ** 0.5).unitary(),
        np.array(
            [
                [1j, 1],
                [1, 1j],
            ]
        )
        * np.sqrt(0.5),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.Y(a) ** 0.25).unitary(), cirq.unitary(cirq.Y(a) ** 0.25), atol=1e-8
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.Z(a), cirq.X(b)).unitary(),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, -1],
                [0, 0, -1, 0],
            ]
        ),
        atol=1e-8,
    )

    # Single qubit gates and two qubit gate.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.Z(a), cirq.X(b), cirq.CNOT(a, b)).unitary(),
        np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1],
            ]
        ),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.H(b), cirq.CNOT(b, a) ** 0.5, cirq.Y(a) ** 0.5).unitary(),
        np.array(
            [
                [1, 1, -1, -1],
                [1j, -1j, -1j, 1j],
                [1, 1, 1, 1],
                [1, -1, 1, -1],
            ]
        )
        * np.sqrt(0.25),
        atol=1e-8,
    )

    # Measurement gate has no corresponding matrix.
    c = circuit_cls(cirq.measure(a))
    with pytest.raises(ValueError):
        _ = c.unitary(ignore_terminal_measurements=False)

    # Ignoring terminal measurements.
    c = circuit_cls(cirq.measure(a))
    cirq.testing.assert_allclose_up_to_global_phase(c.unitary(), np.eye(2), atol=1e-8)

    # Ignoring terminal measurements with further cirq.
    c = circuit_cls(cirq.Z(a), cirq.measure(a), cirq.Z(b))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.unitary(), np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]), atol=1e-8
    )

    # Optionally don't ignoring terminal measurements.
    c = circuit_cls(cirq.measure(a))
    with pytest.raises(ValueError, match="measurement"):
        _ = (c.unitary(ignore_terminal_measurements=False),)

    # Non-terminal measurements are not ignored.
    c = circuit_cls(cirq.measure(a), cirq.X(a))
    with pytest.raises(ValueError):
        _ = c.unitary()

    # Non-terminal measurements are not ignored (multiple qubits).
    c = circuit_cls(cirq.measure(a), cirq.measure(b), cirq.CNOT(a, b))
    with pytest.raises(ValueError):
        _ = c.unitary()

    # Gates without matrix or decomposition raise exception
    class MysteryGate(cirq.testing.TwoQubitGate):
        pass

    c = circuit_cls(MysteryGate()(a, b))
    with pytest.raises(TypeError):
        _ = c.unitary()

    # Accounts for measurement bit flipping.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.measure(a, invert_mask=(True,))).unitary(), cirq.unitary(cirq.X), atol=1e-8
    )

    # dtype
    c = circuit_cls(cirq.X(a))
    assert c.unitary(dtype=np.complex64).dtype == np.complex64
    assert c.unitary(dtype=np.complex128).dtype == np.complex128
    assert c.unitary(dtype=np.float64).dtype == np.float64


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_circuit_unitary(circuit_cls):
    q = cirq.NamedQubit('q')

    with_inner_measure = circuit_cls(cirq.H(q), cirq.measure(q), cirq.H(q))
    assert not cirq.has_unitary(with_inner_measure)
    assert cirq.unitary(with_inner_measure, None) is None

    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(
            circuit_cls(cirq.X(q) ** 0.5),
            cirq.measure(q),
        ),
        np.array(
            [
                [1j, 1],
                [1, 1j],
            ]
        )
        * np.sqrt(0.5),
        atol=1e-8,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_simple_circuits_to_unitary_matrix(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Phase parity.
    c = circuit_cls(cirq.CNOT(a, b), cirq.Z(b), cirq.CNOT(a, b))
    assert cirq.has_unitary(c)
    m = c.unitary()
    cirq.testing.assert_allclose_up_to_global_phase(
        m,
        np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ]
        ),
        atol=1e-8,
    )

    # 2-qubit matrix matches when qubits in order.
    for expected in [np.diag([1, 1j, -1, -1j]), cirq.unitary(cirq.CNOT)]:

        class Passthrough(cirq.testing.TwoQubitGate):
            def _unitary_(self) -> np.ndarray:
                return expected

        c = circuit_cls(Passthrough()(a, b))
        m = c.unitary()
        cirq.testing.assert_allclose_up_to_global_phase(m, expected, atol=1e-8)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_composite_gate_to_unitary_matrix(circuit_cls):
    class CnotComposite(cirq.testing.TwoQubitGate):
        def _decompose_(self, qubits):
            q0, q1 = qubits
            return cirq.Y(q1) ** -0.5, cirq.CZ(q0, q1), cirq.Y(q1) ** 0.5

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = circuit_cls(
        cirq.X(a), CnotComposite()(a, b), cirq.X(a), cirq.measure(a), cirq.X(b), cirq.measure(b)
    )
    assert cirq.has_unitary(c)

    mat = c.unitary()
    mat_expected = cirq.unitary(cirq.CNOT)

    cirq.testing.assert_allclose_up_to_global_phase(mat, mat_expected, atol=1e-8)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_expanding_gate_symbols(circuit_cls):
    class MultiTargetCZ(cirq.Gate):
        def __init__(self, num_qubits):
            self._num_qubits = num_qubits

        def num_qubits(self) -> int:
            return self._num_qubits

        def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> Tuple[str, ...]:
            assert args.known_qubit_count is not None
            return ('@',) + ('Z',) * (args.known_qubit_count - 1)

    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    t0 = circuit_cls(MultiTargetCZ(1).on(c))
    t1 = circuit_cls(MultiTargetCZ(2).on(c, a))
    t2 = circuit_cls(MultiTargetCZ(3).on(c, a, b))

    cirq.testing.assert_has_diagram(
        t0,
        """
c: ───@───
""",
    )

    cirq.testing.assert_has_diagram(
        t1,
        """
a: ───Z───
      │
c: ───@───
""",
    )

    cirq.testing.assert_has_diagram(
        t2,
        """
a: ───Z───
      │
b: ───Z───
      │
c: ───@───
""",
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_transposed_diagram_exponent_order(circuit_cls):
    a, b, c = cirq.LineQubit.range(3)
    circuit = circuit_cls(
        cirq.CZ(a, b) ** -0.5,
        cirq.CZ(a, c) ** 0.5,
        cirq.CZ(b, c) ** 0.125,
    )
    cirq.testing.assert_has_diagram(
        circuit,
        """
0 1      2
│ │      │
@─@^-0.5 │
│ │      │
@─┼──────@^0.5
│ │      │
│ @──────@^(1/8)
│ │      │
""",
        transpose=True,
    )


def test_insert_moments():
    q = cirq.NamedQubit('q')
    c = cirq.Circuit()

    m0 = cirq.Moment([cirq.X(q)])
    c.append(m0)
    assert list(c) == [m0]
    assert c[0] == m0

    m1 = cirq.Moment([cirq.Y(q)])
    c.append(m1)
    assert list(c) == [m0, m1]
    assert c[1] == m1

    m2 = cirq.Moment([cirq.Z(q)])
    c.insert(0, m2)
    assert list(c) == [m2, m0, m1]
    assert c[0] == m2

    assert c._moments == [m2, m0, m1]
    assert c._moments[0] == m2


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_apply_unitary_effect_to_state(circuit_cls):
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # State ordering.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a) ** 0.5).final_state_vector(),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=0),
        np.array([1j, 1]) * np.sqrt(0.5),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a) ** 0.5).final_state_vector(initial_state=1),
        np.array([1, 1j]) * np.sqrt(0.5),
        atol=1e-8,
    )

    # Vector state.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a) ** 0.5).final_state_vector(
            initial_state=np.array([1j, 1]) * np.sqrt(0.5)
        ),
        np.array([0, 1]),
        atol=1e-8,
    )

    # Qubit ordering.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=0),
        np.array([1, 0, 0, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=1),
        np.array([0, 1, 0, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=2),
        np.array([0, 0, 0, 1]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(initial_state=3),
        np.array([0, 0, 1, 0]),
        atol=1e-8,
    )

    # Product state
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(
            initial_state=cirq.KET_ZERO(a) * cirq.KET_ZERO(b)
        ),
        np.array([1, 0, 0, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(
            initial_state=cirq.KET_ZERO(a) * cirq.KET_ONE(b)
        ),
        np.array([0, 1, 0, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(
            initial_state=cirq.KET_ONE(a) * cirq.KET_ZERO(b)
        ),
        np.array([0, 0, 0, 1]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.CNOT(a, b)).final_state_vector(
            initial_state=cirq.KET_ONE(a) * cirq.KET_ONE(b)
        ),
        np.array([0, 0, 1, 0]),
        atol=1e-8,
    )

    # Measurements.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.measure(a)).final_state_vector(), np.array([1, 0]), atol=1e-8
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(a), cirq.measure(a)).final_state_vector(), np.array([0, 1]), atol=1e-8
    )
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(
            circuit_cls(cirq.measure(a), cirq.X(a)).final_state_vector(),
            np.array([1, 0]),
            atol=1e-8,
        )
    with pytest.raises(ValueError):
        cirq.testing.assert_allclose_up_to_global_phase(
            circuit_cls(cirq.measure(a)).final_state_vector(ignore_terminal_measurements=False),
            np.array([1, 0]),
            atol=1e-8,
        )

    # Extra qubits.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls().final_state_vector(), np.array([1]), atol=1e-8
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls().final_state_vector(qubits_that_should_be_present=[a]),
        np.array([1, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.X(b)).final_state_vector(qubits_that_should_be_present=[a]),
        np.array([0, 1, 0, 0]),
        atol=1e-8,
    )

    # Qubit order.
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[a, b]),
        np.array([0, 1, 0, 0]),
        atol=1e-8,
    )
    cirq.testing.assert_allclose_up_to_global_phase(
        circuit_cls(cirq.Z(a), cirq.X(b)).final_state_vector(qubit_order=[b, a]),
        np.array([0, 0, 1, 0]),
        atol=1e-8,
    )

    # Dtypes.
    dtypes = [np.complex64, np.complex128]
    if hasattr(np, 'complex256'):  # Some systems don't support 128 bit floats.
        dtypes.append(np.complex256)
    for dt in dtypes:
        cirq.testing.assert_allclose_up_to_global_phase(
            circuit_cls(cirq.X(a) ** 0.5).final_state_vector(
                initial_state=np.array([1j, 1]) * np.sqrt(0.5), dtype=dt
            ),
            np.array([0, 1]),
            atol=1e-8,
        )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_is_parameterized(circuit_cls, resolve_fn):
    a, b = cirq.LineQubit.range(2)
    circuit = circuit_cls(
        cirq.CZ(a, b) ** sympy.Symbol('u'),
        cirq.X(a) ** sympy.Symbol('v'),
        cirq.Y(b) ** sympy.Symbol('w'),
    )
    assert cirq.is_parameterized(circuit)

    circuit = resolve_fn(circuit, cirq.ParamResolver({'u': 0.1, 'v': 0.3}))
    assert cirq.is_parameterized(circuit)

    circuit = resolve_fn(circuit, cirq.ParamResolver({'w': 0.2}))
    assert not cirq.is_parameterized(circuit)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_resolve_parameters(circuit_cls, resolve_fn):
    a, b = cirq.LineQubit.range(2)
    circuit = circuit_cls(
        cirq.CZ(a, b) ** sympy.Symbol('u'),
        cirq.X(a) ** sympy.Symbol('v'),
        cirq.Y(b) ** sympy.Symbol('w'),
    )
    resolved_circuit = resolve_fn(circuit, cirq.ParamResolver({'u': 0.1, 'v': 0.3, 'w': 0.2}))
    cirq.testing.assert_has_diagram(
        resolved_circuit,
        """
0: ───@───────X^0.3───
      │
1: ───@^0.1───Y^0.2───
""",
    )
    q = cirq.NamedQubit('q')
    # no-op parameter resolution
    circuit = circuit_cls([cirq.Moment(), cirq.Moment([cirq.X(q)])])
    resolved_circuit = resolve_fn(circuit, cirq.ParamResolver({}))
    cirq.testing.assert_same_circuits(circuit, resolved_circuit)
    # actually resolve something
    circuit = circuit_cls([cirq.Moment(), cirq.Moment([cirq.X(q) ** sympy.Symbol('x')])])
    resolved_circuit = resolve_fn(circuit, cirq.ParamResolver({'x': 0.2}))
    expected_circuit = circuit_cls([cirq.Moment(), cirq.Moment([cirq.X(q) ** 0.2])])
    cirq.testing.assert_same_circuits(expected_circuit, resolved_circuit)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
@pytest.mark.parametrize('resolve_fn', [cirq.resolve_parameters, cirq.resolve_parameters_once])
def test_parameter_names(circuit_cls, resolve_fn):
    a, b = cirq.LineQubit.range(2)
    circuit = circuit_cls(
        cirq.CZ(a, b) ** sympy.Symbol('u'),
        cirq.X(a) ** sympy.Symbol('v'),
        cirq.Y(b) ** sympy.Symbol('w'),
    )
    resolved_circuit = resolve_fn(circuit, cirq.ParamResolver({'u': 0.1, 'v': 0.3, 'w': 0.2}))
    assert cirq.parameter_names(circuit) == {'u', 'v', 'w'}
    assert cirq.parameter_names(resolved_circuit) == set()


def test_items():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit()
    m1 = cirq.Moment([cirq.X(a), cirq.X(b)])
    m2 = cirq.Moment([cirq.X(a)])
    m3 = cirq.Moment([])
    m4 = cirq.Moment([cirq.CZ(a, b)])

    c[:] = [m1, m2]
    cirq.testing.assert_same_circuits(c, cirq.Circuit([m1, m2]))

    assert c[0] == m1
    del c[0]
    cirq.testing.assert_same_circuits(c, cirq.Circuit([m2]))

    c.append(m1)
    c.append(m3)
    cirq.testing.assert_same_circuits(c, cirq.Circuit([m2, m1, m3]))

    assert c[0:2] == cirq.Circuit([m2, m1])
    c[0:2] = [m4]
    cirq.testing.assert_same_circuits(c, cirq.Circuit([m4, m3]))

    c[:] = [m1]
    cirq.testing.assert_same_circuits(c, cirq.Circuit([m1]))

    with pytest.raises(TypeError):
        c[:] = [m1, 1]
    with pytest.raises(TypeError):
        c[0] = 1


def test_copy():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.Circuit(cirq.X(a), cirq.CZ(a, b), cirq.Z(a), cirq.Z(b))
    assert c == c.copy() == c.__copy__()
    c2 = c.copy()
    assert c2 == c
    c2[:] = []
    assert c2 != c


def test_batch_remove():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Empty case.
    after = original.copy()
    after.batch_remove([])
    assert after == original

    # Delete one.
    after = original.copy()
    after.batch_remove([(0, cirq.X(a))])
    assert after == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Out of range.
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_remove([(500, cirq.X(a))])
    assert after == original

    # Delete several.
    after = original.copy()
    after.batch_remove([(0, cirq.X(a)), (2, cirq.CZ(a, b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment(),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Delete all.
    after = original.copy()
    after.batch_remove(
        [(0, cirq.X(a)), (1, cirq.Z(b)), (2, cirq.CZ(a, b)), (3, cirq.X(a)), (3, cirq.X(b))]
    )
    assert after == cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment(),
        ]
    )

    # Delete moment partially.
    after = original.copy()
    after.batch_remove([(3, cirq.X(a))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(b)]),
        ]
    )

    # Deleting something that's not there.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(b))])
    assert after == original

    # Duplicate delete.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_remove([(0, cirq.X(a)), (0, cirq.X(a))])
    assert after == original


def test_batch_replace():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Empty case.
    after = original.copy()
    after.batch_replace([])
    assert after == original

    # Replace one.
    after = original.copy()
    after.batch_replace([(0, cirq.X(a), cirq.Y(a))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.Y(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Out of range.
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_replace([(500, cirq.X(a), cirq.Y(a))])
    assert after == original

    # Gate does not exist.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_replace([(0, cirq.Z(a), cirq.Y(a))])
    assert after == original

    # Replace several.
    after = original.copy()
    after.batch_replace([(0, cirq.X(a), cirq.Y(a)), (2, cirq.CZ(a, b), cirq.CNOT(a, b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.Y(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CNOT(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )


def test_batch_insert_into():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    original = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Empty case.
    after = original.copy()
    after.batch_insert_into([])
    assert after == original

    # Add into non-empty moment.
    after = original.copy()
    after.batch_insert_into([(0, cirq.X(b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Add multiple operations into non-empty moment.
    after = original.copy()
    after.batch_insert_into([(0, [cirq.X(b), cirq.X(c)])])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)]),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Add into empty moment.
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Add multiple operations into empty moment.
    after = original.copy()
    after.batch_insert_into([(1, [cirq.Z(a), cirq.Z(b)])])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([cirq.Z(a), cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Add into two moments.
    after = original.copy()
    after.batch_insert_into([(1, cirq.Z(b)), (0, cirq.X(b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.X(b)]),
            cirq.Moment([cirq.Z(b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Out of range.
    after = original.copy()
    with pytest.raises(IndexError):
        after.batch_insert_into([(500, cirq.X(a))])
    assert after == original

    # Collision.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(0, cirq.X(a))])
    assert after == original

    # Collision with multiple operations.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(0, [cirq.X(b), cirq.X(c), cirq.X(a)])])
    assert after == original

    # Duplicate insertion collision.
    after = original.copy()
    with pytest.raises(ValueError):
        after.batch_insert_into([(1, cirq.X(a)), (1, cirq.CZ(a, b))])
    assert after == original


def test_batch_insert():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    original = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )

    # Empty case.
    after = original.copy()
    after.batch_insert([])
    assert after == original

    # Pushing.
    after = original.copy()
    after.batch_insert([(0, cirq.CZ(a, b)), (0, cirq.CNOT(a, b)), (1, cirq.Z(b))])
    assert after == cirq.Circuit(
        [
            cirq.Moment([cirq.CNOT(a, b)]),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.Z(b)]),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(a, b)]),
            cirq.Moment([cirq.X(a), cirq.X(b)]),
        ]
    )


def test_batch_insert_multiple_same_index():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(b)), (0, cirq.Z(a))])
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit(
            [
                cirq.Moment([cirq.Z(a), cirq.Z(b)]),
                cirq.Moment([cirq.Z(a)]),
            ]
        ),
    )


def test_batch_insert_reverses_order_for_same_index_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit()
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.CZ(a, b)), (0, cirq.Z(b))])
    assert c == cirq.Circuit(cirq.Z(b), cirq.CZ(a, b), cirq.Z(a))


def test_batch_insert_maintains_order_despite_multiple_previous_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.H(a))
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(a)), (0, cirq.Z(a)), (1, cirq.CZ(a, b))])
    assert c == cirq.Circuit([cirq.Z(a)] * 3, cirq.H(a), cirq.CZ(a, b))


def test_batch_insert_doesnt_overshift_due_to_previous_shifts():
    a = cirq.NamedQubit('a')
    c = cirq.Circuit([cirq.H(a)] * 3)
    c.batch_insert([(0, cirq.Z(a)), (0, cirq.Z(a)), (1, cirq.X(a)), (2, cirq.Y(a))])
    assert c == cirq.Circuit(
        cirq.Z(a),
        cirq.Z(a),
        cirq.H(a),
        cirq.X(a),
        cirq.H(a),
        cirq.Y(a),
        cirq.H(a),
    )


def test_batch_insert_doesnt_overshift_due_to_inline_inserts():
    a, b = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.SWAP(a, b), cirq.SWAP(a, b), cirq.H(a), cirq.SWAP(a, b), cirq.SWAP(a, b))
    c.batch_insert([(0, cirq.X(a)), (3, cirq.X(b)), (4, cirq.Y(a))])
    assert c == cirq.Circuit(
        cirq.X(a),
        cirq.SWAP(a, b),
        cirq.SWAP(a, b),
        cirq.H(a),
        cirq.X(b),
        cirq.SWAP(a, b),
        cirq.Y(a),
        cirq.SWAP(a, b),
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_next_moments_operating_on(circuit_cls):
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = cirq.testing.random_circuit(randint(1, 20), n_moments, random())
        circuit_qubits = circuit.all_qubits()
        n_key_qubits = randint(int(bool(circuit_qubits)), len(circuit_qubits))
        key_qubits = sample(circuit_qubits, n_key_qubits)
        start = randrange(len(circuit))
        next_moments = circuit.next_moments_operating_on(key_qubits, start)
        for q, m in next_moments.items():
            if m == len(circuit):
                p = circuit.prev_moment_operating_on([q])
            else:
                p = circuit.prev_moment_operating_on([q], m - 1)
            assert (not p) or (p < start)


def test_pick_inserted_ops_moment_indices():
    for _ in range(20):
        n_moments = randint(1, 10)
        n_qubits = randint(1, 20)
        op_density = random()
        circuit = cirq.testing.random_circuit(n_qubits, n_moments, op_density)
        start = randrange(n_moments)
        first_half = cirq.Circuit(circuit[:start])
        second_half = cirq.Circuit(circuit[start:])
        operations = tuple(op for moment in second_half for op in moment.operations)
        squeezed_second_half = cirq.Circuit(operations, strategy=cirq.InsertStrategy.EARLIEST)
        expected_circuit = cirq.Circuit(first_half._moments + squeezed_second_half._moments)
        expected_circuit._moments += [
            cirq.Moment() for _ in range(len(circuit) - len(expected_circuit))
        ]
        insert_indices, _ = circuits.circuit._pick_inserted_ops_moment_indices(operations, start)
        actual_circuit = cirq.Circuit(
            first_half._moments + [cirq.Moment() for _ in range(n_moments - start)]
        )
        for op, insert_index in zip(operations, insert_indices):
            actual_circuit._moments[insert_index] = actual_circuit._moments[
                insert_index
            ].with_operation(op)
        assert actual_circuit == expected_circuit


def test_push_frontier_new_moments():
    operation = cirq.X(cirq.NamedQubit('q'))
    insertion_index = 3
    circuit = cirq.Circuit()
    circuit._insert_operations([operation], [insertion_index])
    assert circuit == cirq.Circuit(
        [cirq.Moment() for _ in range(insertion_index)] + [cirq.Moment([operation])]
    )


def test_push_frontier_random_circuit():
    for _ in range(20):
        n_moments = randint(1, 10)
        circuit = cirq.testing.random_circuit(randint(1, 20), n_moments, random())
        qubits = circuit.all_qubits()
        early_frontier = {q: randint(0, n_moments) for q in sample(qubits, randint(0, len(qubits)))}
        late_frontier = {q: randint(0, n_moments) for q in sample(qubits, randint(0, len(qubits)))}
        update_qubits = sample(qubits, randint(0, len(qubits)))

        orig_early_frontier = {q: f for q, f in early_frontier.items()}
        orig_moments = [m for m in circuit._moments]
        insert_index, n_new_moments = circuit._push_frontier(
            early_frontier, late_frontier, update_qubits
        )

        assert set(early_frontier.keys()) == set(orig_early_frontier.keys())
        for q in set(early_frontier).difference(update_qubits):
            assert early_frontier[q] == orig_early_frontier[q]
        for q, f in late_frontier.items():
            assert orig_early_frontier.get(q, 0) <= late_frontier[q] + n_new_moments
            if f != len(orig_moments):
                assert orig_moments[f] == circuit[f + n_new_moments]
        for q in set(update_qubits).intersection(early_frontier):
            if orig_early_frontier[q] == insert_index:
                assert orig_early_frontier[q] == early_frontier[q]
                assert (not n_new_moments) or (circuit._moments[early_frontier[q]] == cirq.Moment())
            elif orig_early_frontier[q] == len(orig_moments):
                assert early_frontier[q] == len(circuit)
            else:
                assert orig_moments[orig_early_frontier[q]] == circuit._moments[early_frontier[q]]


@pytest.mark.parametrize(
    'circuit', [cirq.testing.random_circuit(cirq.LineQubit.range(10), 10, 0.5) for _ in range(20)]
)
def test_insert_operations_random_circuits(circuit):
    n_moments = len(circuit)
    operations, insert_indices = [], []
    for moment_index, moment in enumerate(circuit):
        for op in moment.operations:
            operations.append(op)
            insert_indices.append(moment_index)
    other_circuit = cirq.Circuit([cirq.Moment() for _ in range(n_moments)])
    other_circuit._insert_operations(operations, insert_indices)
    assert circuit == other_circuit


def test_insert_operations_errors():
    a, b, c = (cirq.NamedQubit(s) for s in 'abc')
    with pytest.raises(ValueError):
        circuit = cirq.Circuit([cirq.Moment([cirq.Z(c)])])
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = [0, 0]
        circuit._insert_operations(operations, insertion_indices)

    with pytest.raises(ValueError):
        circuit = cirq.Circuit(cirq.X(a))
        operations = [cirq.CZ(a, b)]
        insertion_indices = [0]
        circuit._insert_operations(operations, insertion_indices)

    with pytest.raises(ValueError):
        circuit = cirq.Circuit()
        operations = [cirq.X(a), cirq.CZ(a, b)]
        insertion_indices = []
        circuit._insert_operations(operations, insertion_indices)


def test_validates_while_editing():
    c = cirq.Circuit(device=FOXY)

    with pytest.raises(ValueError, match='Unsupported qubit type'):
        # Wrong type of qubit.
        c.append(cirq.Z(cirq.NamedQubit('q')))
    with pytest.raises(ValueError, match='Qubit not on device'):
        # A qubit that's not on the device.
        c[:] = [cirq.Moment([cirq.Z(cirq.GridQubit(-5, 100))])]
    c.append(cirq.Z(cirq.GridQubit(0, 0)))

    with pytest.raises(ValueError, match='Non-local interaction'):
        # Non-adjacent CZ.
        c[0] = cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 1))])

    c.insert(0, cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(1, 0)))


def test_respects_additional_adjacency_constraints():
    c = cirq.Circuit(device=FOXY)
    c.append(cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1)))
    c.append(
        cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)), strategy=cirq.InsertStrategy.EARLIEST
    )
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))]),
                cirq.Moment([cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))]),
            ],
            device=FOXY,
        ),
    )


def test_commutes_past_adjacency_constraints():
    c = cirq.Circuit(
        [
            cirq.Moment(),
            cirq.Moment(),
            cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))]),
        ],
        device=FOXY,
    )
    c.append(
        cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1)), strategy=cirq.InsertStrategy.EARLIEST
    )
    cirq.testing.assert_same_circuits(
        c,
        cirq.Circuit(
            [
                cirq.Moment([cirq.CZ(cirq.GridQubit(1, 0), cirq.GridQubit(1, 1))]),
                cirq.Moment(),
                cirq.Moment([cirq.CZ(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))]),
            ],
            device=FOXY,
        ),
    )


def test_decomposes_while_appending():
    c = cirq.Circuit(device=FOXY)
    c.append(cirq.TOFFOLI(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)))
    cirq.testing.assert_allclose_up_to_global_phase(
        c.unitary(), cirq.unitary(cirq.TOFFOLI), atol=1e-8
    )

    # But you still have to respect adjacency constraints!
    with pytest.raises(ValueError):
        c.append(cirq.TOFFOLI(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(0, 6)))


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_to_qasm(circuit_cls):
    q0 = cirq.NamedQubit('q0')
    circuit = circuit_cls(
        cirq.X(q0),
    )
    assert circuit.to_qasm() == cirq.qasm(circuit)
    assert (
        circuit.to_qasm()
        == f"""// Generated from Cirq v{cirq.__version__}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
"""
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_save_qasm(tmpdir, circuit_cls):
    file_path = os.path.join(tmpdir, 'test.qasm')
    q0 = cirq.NamedQubit('q0')
    circuit = circuit_cls(
        cirq.X(q0),
    )

    circuit.save_qasm(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
    assert (
        file_content
        == f"""// Generated from Cirq v{cirq.__version__}

OPENQASM 2.0;
include "qelib1.inc";


// Qubits: [q0]
qreg q[1];


x q[0];
"""
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_findall_operations_between(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)

    #    0: ───H───@───────────────────────────────────────@───H───
    #              │                                       │
    #    1: ───────@───H───@───────────────────────@───H───@───────
    #                      │                       │
    #    2: ───────────────@───H───@───────@───H───@───────────────
    #                              │       │
    #    3: ───────────────────────@───H───@───────────────────────
    #
    # moments: 0   1   2   3   4   5   6   7   8   9   10  11  12
    circuit = circuit_cls(
        cirq.H(a),
        cirq.CZ(a, b),
        cirq.H(b),
        cirq.CZ(b, c),
        cirq.H(c),
        cirq.CZ(c, d),
        cirq.H(d),
        cirq.CZ(c, d),
        cirq.H(c),
        cirq.CZ(b, c),
        cirq.H(b),
        cirq.CZ(a, b),
        cirq.H(a),
    )

    # Empty frontiers means no results.
    actual = circuit.findall_operations_between(start_frontier={}, end_frontier={})
    assert actual == []

    # Empty range is empty.
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={a: 5})
    assert actual == []

    # Default end_frontier value is len(circuit.
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={})
    assert actual == [
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Default start_frontier value is 0.
    actual = circuit.findall_operations_between(start_frontier={}, end_frontier={a: 5})
    assert actual == [(0, cirq.H(a)), (1, cirq.CZ(a, b))]

    # omit_crossing_operations omits crossing operations.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5}, end_frontier={}, omit_crossing_operations=True
    )
    assert actual == [
        (12, cirq.H(a)),
    ]

    # omit_crossing_operations keeps operations across included regions.
    actual = circuit.findall_operations_between(
        start_frontier={a: 5, b: 5}, end_frontier={}, omit_crossing_operations=True
    )
    assert actual == [
        (10, cirq.H(b)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together.
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={b: 5})
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]

    # Regions are OR'd together, not AND'd together (2).
    actual = circuit.findall_operations_between(start_frontier={a: 5}, end_frontier={a: 5, b: 5})
    assert actual == [
        (1, cirq.CZ(a, b)),
        (2, cirq.H(b)),
        (3, cirq.CZ(b, c)),
    ]

    # Inclusive start, exclusive end.
    actual = circuit.findall_operations_between(start_frontier={c: 4}, end_frontier={c: 8})
    assert actual == [
        (4, cirq.H(c)),
        (5, cirq.CZ(c, d)),
        (7, cirq.CZ(c, d)),
    ]

    # Out of range is clamped.
    actual = circuit.findall_operations_between(start_frontier={a: -100}, end_frontier={a: +100})
    assert actual == [
        (0, cirq.H(a)),
        (1, cirq.CZ(a, b)),
        (11, cirq.CZ(a, b)),
        (12, cirq.H(a)),
    ]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_reachable_frontier_from(circuit_cls):
    a, b, c, d = cirq.LineQubit.range(4)

    #    0: ───H───@───────────────────────────────────────@───H───
    #              │                                       │
    #    1: ───────@───H───@───────────────────────@───H───@───────
    #                      │                       │
    #    2: ───────────────@───H───@───────@───H───@───────────────
    #                              │       │
    #    3: ───────────────────────@───H───@───────────────────────
    #
    # moments: 0   1   2   3   4   5   6   7   8   9   10  11  12
    circuit = circuit_cls(
        cirq.H(a),
        cirq.CZ(a, b),
        cirq.H(b),
        cirq.CZ(b, c),
        cirq.H(c),
        cirq.CZ(c, d),
        cirq.H(d),
        cirq.CZ(c, d),
        cirq.H(c),
        cirq.CZ(b, c),
        cirq.H(b),
        cirq.CZ(a, b),
        cirq.H(a),
    )

    # Empty cases.
    assert circuit_cls().reachable_frontier_from(start_frontier={}) == {}
    assert circuit.reachable_frontier_from(start_frontier={}) == {}

    # Clamped input cases.
    assert circuit_cls().reachable_frontier_from(start_frontier={a: 5}) == {a: 5}
    assert circuit_cls().reachable_frontier_from(start_frontier={a: -100}) == {a: 0}
    assert circuit.reachable_frontier_from(start_frontier={a: 100}) == {a: 100}

    # Stopped by crossing outside case.
    assert circuit.reachable_frontier_from({a: -1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 0}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 1}) == {a: 1}
    assert circuit.reachable_frontier_from({a: 2}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 5}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 10}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 11}) == {a: 11}
    assert circuit.reachable_frontier_from({a: 12}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 13}) == {a: 13}
    assert circuit.reachable_frontier_from({a: 14}) == {a: 14}

    # Inside crossing works only before blocked case.
    assert circuit.reachable_frontier_from({a: 0, b: 0}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 2, b: 2}) == {a: 11, b: 3}
    assert circuit.reachable_frontier_from({a: 0, b: 4}) == {a: 1, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 4}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 9}) == {a: 11, b: 9}
    assert circuit.reachable_frontier_from({a: 3, b: 10}) == {a: 13, b: 13}

    # Travelling shadow.
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0}) == {a: 11, b: 9, c: 5}

    # Full circuit
    assert circuit.reachable_frontier_from({a: 0, b: 0, c: 0, d: 0}) == {a: 13, b: 13, c: 13, d: 13}

    # Blocker.
    assert circuit.reachable_frontier_from(
        {a: 0, b: 0, c: 0, d: 0}, is_blocker=lambda op: op == cirq.CZ(b, c)
    ) == {a: 11, b: 3, c: 3, d: 5}


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_submoments(circuit_cls):
    a, b, c, d, e, f = cirq.LineQubit.range(6)
    circuit = circuit_cls(
        cirq.H.on(a),
        cirq.H.on(d),
        cirq.CZ.on(a, d),
        cirq.CZ.on(b, c),
        (cirq.CNOT ** 0.5).on(a, d),
        (cirq.CNOT ** 0.5).on(b, e),
        (cirq.CNOT ** 0.5).on(c, f),
        cirq.H.on(c),
        cirq.H.on(e),
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
          ┌───────────┐   ┌──────┐
0: ───H────@───────────────@─────────
           │               │
1: ───@────┼@──────────────┼─────────
      │    ││              │
2: ───@────┼┼────@─────────┼────H────
           ││    │         │
3: ───H────@┼────┼─────────X^0.5─────
            │    │
4: ─────────X^0.5┼─────────H─────────
                 │
5: ──────────────X^0.5───────────────
          └───────────┘   └──────┘
""",
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
  0 1 2 3     4     5
  │ │ │ │     │     │
  H @─@ H     │     │
  │ │ │ │     │     │
┌╴│ │ │ │     │     │    ╶┐
│ @─┼─┼─@     │     │     │
│ │ @─┼─┼─────X^0.5 │     │
│ │ │ @─┼─────┼─────X^0.5 │
└╴│ │ │ │     │     │    ╶┘
  │ │ │ │     │     │
┌╴│ │ │ │     │     │    ╶┐
│ @─┼─┼─X^0.5 H     │     │
│ │ │ H │     │     │     │
└╴│ │ │ │     │     │    ╶┘
  │ │ │ │     │     │
""",
        transpose=True,
    )

    cirq.testing.assert_has_diagram(
        circuit,
        r"""
          /-----------\   /------\
0: ---H----@---------------@---------
           |               |
1: ---@----|@--------------|---------
      |    ||              |
2: ---@----||----@---------|----H----
           ||    |         |
3: ---H----@|----|---------X^0.5-----
            |    |
4: ---------X^0.5|---------H---------
                 |
5: --------------X^0.5---------------
          \-----------/   \------/
""",
        use_unicode_characters=False,
    )

    cirq.testing.assert_has_diagram(
        circuit,
        r"""
  0 1 2 3     4     5
  | | | |     |     |
  H @-@ H     |     |
  | | | |     |     |
/ | | | |     |     |     \
| @-----@     |     |     |
| | @---------X^0.5 |     |
| | | @-------------X^0.5 |
\ | | | |     |     |     /
  | | | |     |     |
/ | | | |     |     |     \
| @-----X^0.5 H     |     |
| | | H |     |     |     |
\ | | | |     |     |     /
  | | | |     |     |
""",
        use_unicode_characters=False,
        transpose=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_decompose(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose(circuit_cls(cirq.X(a), cirq.Y(b), cirq.CZ(a, b))) == [
        cirq.X(a),
        cirq.Y(b),
        cirq.CZ(a, b),
    ]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_measurement_key_mapping(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    c = circuit_cls(
        cirq.X(a),
        cirq.measure(a, key='m1'),
        cirq.measure(b, key='m2'),
    )
    assert c.all_measurement_key_names() == {'m1', 'm2'}

    assert cirq.with_measurement_key_mapping(c, {'m1': 'p1'}).all_measurement_key_names() == {
        'p1',
        'm2',
    }

    assert cirq.with_measurement_key_mapping(
        c, {'m1': 'p1', 'm2': 'p2'}
    ).all_measurement_key_names() == {'p1', 'p2'}

    c_swapped = cirq.with_measurement_key_mapping(c, {'m1': 'm2', 'm2': 'm1'})
    assert c_swapped.all_measurement_key_names() == {'m1', 'm2'}

    # Verify that the keys were actually swapped.
    simulator = cirq.Simulator()
    assert simulator.run(c).measurements == {'m1': 1, 'm2': 0}
    assert simulator.run(c_swapped).measurements == {'m1': 0, 'm2': 1}

    assert (
        cirq.with_measurement_key_mapping(
            c,
            {
                'x': 'z',
            },
        ).all_measurement_key_names()
        == {'m1', 'm2'}
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_measurement_key_mapping_preserves_moments(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    c = circuit_cls(
        cirq.Moment(cirq.X(a)),
        cirq.Moment(),
        cirq.Moment(cirq.measure(a, key='m1')),
        cirq.Moment(cirq.measure(b, key='m2')),
    )

    key_map = {'m1': 'p1'}
    remapped_circuit = cirq.with_measurement_key_mapping(c, key_map)
    assert list(remapped_circuit.moments) == [
        cirq.with_measurement_key_mapping(moment, key_map) for moment in c.moments
    ]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_inverse(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    forward = circuit_cls((cirq.X ** 0.5)(a), (cirq.Y ** -0.2)(b), cirq.CZ(a, b))
    backward = circuit_cls((cirq.CZ ** (-1.0))(a, b), (cirq.X ** (-0.5))(a), (cirq.Y ** (0.2))(b))
    cirq.testing.assert_same_circuits(cirq.inverse(forward), backward)

    cirq.testing.assert_same_circuits(cirq.inverse(circuit_cls()), circuit_cls())

    no_inverse = circuit_cls(cirq.measure(a, b))
    with pytest.raises(TypeError, match='__pow__'):
        cirq.inverse(no_inverse)

    # Default when there is no inverse for an op.
    default = circuit_cls((cirq.X ** 0.5)(a), (cirq.Y ** -0.2)(b))
    cirq.testing.assert_same_circuits(cirq.inverse(no_inverse, default), default)
    assert cirq.inverse(no_inverse, None) is None


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_pow_valid_only_for_minus_1(circuit_cls):
    a, b = cirq.LineQubit.range(2)
    forward = circuit_cls((cirq.X ** 0.5)(a), (cirq.Y ** -0.2)(b), cirq.CZ(a, b))

    backward = circuit_cls((cirq.CZ ** (-1.0))(a, b), (cirq.X ** (-0.5))(a), (cirq.Y ** (0.2))(b))

    cirq.testing.assert_same_circuits(cirq.pow(forward, -1), backward)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 1)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, 0)
    with pytest.raises(TypeError, match='__pow__'):
        cirq.pow(forward, -2.5)


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_device_propagates(circuit_cls):
    c = circuit_cls(device=moment_and_op_type_validating_device)
    assert c[:].device is moment_and_op_type_validating_device


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_moment_groups(circuit_cls):
    qubits = [cirq.GridQubit(x, y) for x in range(8) for y in range(8)]
    c0 = cirq.H(qubits[0])
    c7 = cirq.H(qubits[7])
    cz14 = cirq.CZ(qubits[1], qubits[4])
    cz25 = cirq.CZ(qubits[2], qubits[5])
    cz36 = cirq.CZ(qubits[3], qubits[6])
    moment1 = cirq.Moment([c0, cz14, cz25, c7])
    moment2 = cirq.Moment([c0, cz14, cz25, cz36, c7])
    moment3 = cirq.Moment([cz14, cz25, cz36])
    moment4 = cirq.Moment([cz25, cz36])
    circuit = circuit_cls((moment1, moment2, moment3, moment4))
    cirq.testing.assert_has_diagram(
        circuit,
        r"""
           ┌──┐   ┌───┐   ┌───┐   ┌──┐
(0, 0): ────H──────H─────────────────────

(0, 1): ────@──────@───────@─────────────
            │      │       │
(0, 2): ────┼@─────┼@──────┼@──────@─────
            ││     ││      ││      │
(0, 3): ────┼┼─────┼┼@─────┼┼@─────┼@────
            ││     │││     │││     ││
(0, 4): ────@┼─────@┼┼─────@┼┼─────┼┼────
             │      ││      ││     ││
(0, 5): ─────@──────@┼──────@┼─────@┼────
                     │       │      │
(0, 6): ─────────────@───────@──────@────

(0, 7): ────H──────H─────────────────────
           └──┘   └───┘   └───┘   └──┘
""",
        use_unicode_characters=True,
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_moments_property(circuit_cls):
    q = cirq.NamedQubit('q')
    c = circuit_cls(cirq.X(q), cirq.Y(q))
    assert c.moments[0] == cirq.Moment([cirq.X(q)])
    assert c.moments[1] == cirq.Moment([cirq.Y(q)])


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_operation_shape_validation(circuit_cls):
    class BadOperation1(cirq.Operation):
        def _qid_shape_(self):
            return (1,)

        @property
        def qubits(self):
            return cirq.LineQid.for_qid_shape((1, 2, 3))

        def with_qubits(self, *qubits):
            raise NotImplementedError

    class BadOperation2(cirq.Operation):
        def _qid_shape_(self):
            return (1, 2, 3, 9)

        @property
        def qubits(self):
            return cirq.LineQid.for_qid_shape((1, 2, 3))

        def with_qubits(self, *qubits):
            raise NotImplementedError

    _ = circuit_cls(cirq.X(cirq.LineQid(0, 2)))  # Valid
    with pytest.raises(ValueError, match='Invalid operation'):
        _ = circuit_cls(BadOperation1())
    with pytest.raises(ValueError, match='Invalid operation'):
        _ = circuit_cls(BadOperation2())


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_json_dict(circuit_cls):
    q0, q1 = cirq.LineQubit.range(2)
    c = circuit_cls(cirq.CNOT(q0, q1))
    moments = [cirq.Moment([cirq.CNOT(q0, q1)])]
    if circuit_cls == cirq.FrozenCircuit:
        moments = tuple(moments)
    assert c._json_dict_() == {
        'cirq_type': circuit_cls.__name__,
        'moments': moments,
        'device': cirq.UNCONSTRAINED_DEVICE,
    }


def test_with_noise():
    class Noise(cirq.NoiseModel):
        def noisy_operation(self, operation):
            yield operation
            if cirq.LineQubit(0) in operation.qubits:
                yield cirq.H(cirq.LineQubit(0))

    q0, q1 = cirq.LineQubit.range(2)
    c = cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q1),
        cirq.Z(q1),
        cirq.Moment([cirq.X(q0)]),
    )
    c_expected = cirq.Circuit(
        [
            cirq.Moment(
                [
                    cirq.X(q0),
                    cirq.Y(q1),
                ]
            ),
            cirq.Moment(
                [
                    cirq.H(q0),
                ]
            ),
            cirq.Moment(
                [
                    cirq.Z(q1),
                ]
            ),
            cirq.Moment(
                [
                    cirq.X(q0),
                ]
            ),
            cirq.Moment(
                [
                    cirq.H(q0),
                ]
            ),
        ]
    )
    c_noisy = c.with_noise(Noise())
    assert c_noisy == c_expected

    # Accepts NOISE_MODEL_LIKE.
    assert c.with_noise(None) == c
    assert c.with_noise(cirq.depolarize(0.1)) == cirq.Circuit(
        cirq.X(q0),
        cirq.Y(q1),
        cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]),
        cirq.Z(q1),
        cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]),
        cirq.Moment([cirq.X(q0)]),
        cirq.Moment([d.with_tags(ops.VirtualTag()) for d in cirq.depolarize(0.1).on_each(q0, q1)]),
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_init_contents(circuit_cls):
    a, b = cirq.LineQubit.range(2)

    # Moments are not subject to insertion rules.
    c = circuit_cls(
        cirq.Moment([cirq.H(a)]),
        cirq.Moment([cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )
    assert len(c.moments) == 3

    # Earliest packing by default.
    c = circuit_cls(
        cirq.H(a),
        cirq.X(b),
        cirq.CNOT(a, b),
    )
    assert c == circuit_cls(
        cirq.Moment([cirq.H(a), cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )

    # Packing can be controlled.
    c = circuit_cls(cirq.H(a), cirq.X(b), cirq.CNOT(a, b), strategy=cirq.InsertStrategy.NEW)
    assert c == circuit_cls(
        cirq.Moment([cirq.H(a)]),
        cirq.Moment([cirq.X(b)]),
        cirq.Moment([cirq.CNOT(a, b)]),
    )

    circuit_cls()


def test_transform_qubits():
    a, b, c = cirq.LineQubit.range(3)
    original = cirq.Circuit(
        cirq.X(a), cirq.CNOT(a, b), cirq.Moment(), cirq.Moment([cirq.CNOT(b, c)])
    )
    x, y, z = cirq.GridQubit.rect(3, 1, 10, 20)
    desired = cirq.Circuit(
        cirq.X(x), cirq.CNOT(x, y), cirq.Moment(), cirq.Moment([cirq.CNOT(y, z)])
    )
    assert original.transform_qubits(lambda q: cirq.GridQubit(10 + q.x, 20)) == desired
    assert (
        original.transform_qubits(
            {
                a: cirq.GridQubit(10 + a.x, 20),
                b: cirq.GridQubit(10 + b.x, 20),
                c: cirq.GridQubit(10 + c.x, 20),
            }
        )
        == desired
    )
    with pytest.raises(TypeError, match='must be a function or dict'):
        _ = original.transform_qubits('bad arg')

    # Device
    original = cirq.Circuit(device=FOXY)
    assert original.transform_qubits(lambda q: q).device is FOXY
    assert original.transform_qubits(lambda q: q, new_device=BCONE).device is BCONE


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_indexing_by_pair(circuit_cls):
    # 0: ───H───@───X───@───
    #           │       │
    # 1: ───────H───@───@───
    #               │   │
    # 2: ───────────H───X───
    q = cirq.LineQubit.range(3)
    c = circuit_cls(
        [
            cirq.H(q[0]),
            cirq.H(q[1]).controlled_by(q[0]),
            cirq.H(q[2]).controlled_by(q[1]),
            cirq.X(q[0]),
            cirq.CCNOT(*q),
        ]
    )

    # Indexing by single moment and qubit.
    assert c[0, q[0]] == c[0][q[0]] == cirq.H(q[0])
    assert c[1, q[0]] == c[1, q[1]] == cirq.H(q[1]).controlled_by(q[0])
    assert c[2, q[0]] == c[2][q[0]] == cirq.X(q[0])
    assert c[2, q[1]] == c[2, q[2]] == cirq.H(q[2]).controlled_by(q[1])
    assert c[3, q[0]] == c[3, q[1]] == c[3, q[2]] == cirq.CCNOT(*q)

    # Indexing by moment and qubit - throws if there is no operation.
    with pytest.raises(KeyError, match="Moment doesn't act on given qubit"):
        _ = c[0, q[1]]

    # Indexing by single moment and multiple qubits.
    assert c[0, q] == c[0]
    assert c[1, q] == c[1]
    assert c[2, q] == c[2]
    assert c[3, q] == c[3]
    assert c[0, q[0:2]] == c[0]
    assert c[0, q[1:3]] == cirq.Moment([])
    assert c[1, q[1:2]] == c[1]
    assert c[2, [q[0]]] == cirq.Moment([cirq.X(q[0])])
    assert c[2, q[1:3]] == cirq.Moment([cirq.H(q[2]).controlled_by(q[1])])
    assert c[np.int64(2), q[0:2]] == c[2]

    # Indexing by single qubit.
    assert c[:, q[0]] == circuit_cls(
        [
            cirq.Moment([cirq.H(q[0])]),
            cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
            cirq.Moment([cirq.X(q[0])]),
            cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
        ]
    )
    assert c[:, q[1]] == circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
            cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
            cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
        ]
    )
    assert c[:, q[2]] == circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([]),
            cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
            cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
        ]
    )

    # Indexing by several qubits.
    assert c[:, q] == c[:, q[0:2]] == c[:, [q[0], q[2]]] == c
    assert c[:, q[1:3]] == circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
            cirq.Moment([cirq.H(q[2]).controlled_by(q[1])]),
            cirq.Moment([cirq.CCNOT(q[0], q[1], q[2])]),
        ]
    )

    # Indexing by several moments and one qubit.
    assert c[1:3, q[0]] == circuit_cls(
        [
            cirq.H(q[1]).controlled_by(q[0]),
            cirq.X(q[0]),
        ]
    )
    assert c[1::2, q[2]] == circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.CCNOT(*q)]),
        ]
    )

    # Indexing by several moments and several qubits.
    assert c[0:2, q[1:3]] == circuit_cls(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.H(q[1]).controlled_by(q[0])]),
        ]
    )
    assert c[::2, q[0:2]] == circuit_cls(
        [
            cirq.Moment([cirq.H(q[0])]),
            cirq.Moment([cirq.H(q[2]).controlled_by(q[1]), cirq.X(q[0])]),
        ]
    )

    # Equivalent ways of indexing.
    assert c[0:2, q[1:3]] == c[0:2][:, q[1:3]] == c[:, q[1:3]][0:2]

    # Passing more than 2 items is forbidden.
    with pytest.raises(ValueError, match='If key is tuple, it must be a pair.'):
        _ = c[0, q[1], 0]

    # Can't swap indices.
    with pytest.raises(TypeError, match='indices must be integers or slices'):
        _ = c[q[1], 0]


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_indexing_by_numpy_integer(circuit_cls):
    q = cirq.NamedQubit('q')
    c = circuit_cls(cirq.X(q), cirq.Y(q))

    assert c[np.int32(1)] == cirq.Moment([cirq.Y(q)])
    assert c[np.int64(1)] == cirq.Moment([cirq.Y(q)])


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_all_measurement_key_names(circuit_cls):
    class Unknown(cirq.SingleQubitGate):
        def _measurement_key_name_(self):
            return 'test'

    a, b = cirq.LineQubit.range(2)
    c = circuit_cls(
        cirq.X(a),
        cirq.CNOT(a, b),
        cirq.measure(a, key='x'),
        cirq.measure(b, key='y'),
        cirq.reset(a),
        cirq.measure(a, b, key='xy'),
        Unknown().on(a),
    )

    # Big case.
    assert c.all_measurement_key_names() == {'x', 'y', 'xy', 'test'}
    assert c.all_measurement_key_names() == cirq.measurement_key_names(c)
    assert c.all_measurement_key_names() == c.all_measurement_key_objs()

    # Empty case.
    assert circuit_cls().all_measurement_key_names() == set()

    # Order does not matter.
    assert (
        circuit_cls(
            cirq.Moment(
                [
                    cirq.measure(a, key='x'),
                    cirq.measure(b, key='y'),
                ]
            )
        ).all_measurement_key_names()
        == {'x', 'y'}
    )
    assert (
        circuit_cls(
            cirq.Moment(
                [
                    cirq.measure(b, key='y'),
                    cirq.measure(a, key='x'),
                ]
            )
        ).all_measurement_key_names()
        == {'x', 'y'}
    )


def test_zip():
    a, b, c, d = cirq.LineQubit.range(4)

    circuit1 = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
    circuit2 = cirq.Circuit(cirq.X(c), cirq.Y(c), cirq.Z(c))
    circuit3 = cirq.Circuit(cirq.Moment(), cirq.Moment(cirq.S(d)))

    # Calling works both static-style and instance-style.
    assert circuit1.zip(circuit2) == cirq.Circuit.zip(circuit1, circuit2)

    # Empty cases.
    assert cirq.Circuit.zip() == cirq.Circuit()
    assert cirq.Circuit.zip(cirq.Circuit()) == cirq.Circuit()
    assert cirq.Circuit().zip(cirq.Circuit()) == cirq.Circuit()
    assert circuit1.zip(cirq.Circuit()) == circuit1
    assert cirq.Circuit(cirq.Moment()).zip(cirq.Circuit()) == cirq.Circuit(cirq.Moment())
    assert cirq.Circuit().zip(cirq.Circuit(cirq.Moment())) == cirq.Circuit(cirq.Moment())

    # Small cases.
    assert (
        circuit1.zip(circuit2)
        == circuit2.zip(circuit1)
        == cirq.Circuit(
            cirq.Moment(
                cirq.H(a),
                cirq.X(c),
            ),
            cirq.Moment(
                cirq.CNOT(a, b),
                cirq.Y(c),
            ),
            cirq.Moment(
                cirq.Z(c),
            ),
        )
    )
    assert circuit1.zip(circuit2, circuit3) == cirq.Circuit(
        cirq.Moment(
            cirq.H(a),
            cirq.X(c),
        ),
        cirq.Moment(
            cirq.CNOT(a, b),
            cirq.Y(c),
            cirq.S(d),
        ),
        cirq.Moment(
            cirq.Z(c),
        ),
    )

    # Overlapping operations.
    with pytest.raises(ValueError, match="moment index 1.*\n.*CNOT"):
        _ = cirq.Circuit.zip(
            cirq.Circuit(cirq.X(a), cirq.CNOT(a, b)),
            cirq.Circuit(cirq.X(b), cirq.Z(b)),
        )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_zip_alignment(circuit_cls):
    a, b, c = cirq.LineQubit.range(3)

    circuit1 = circuit_cls([cirq.H(a)] * 5)
    circuit2 = circuit_cls([cirq.H(b)] * 3)
    circuit3 = circuit_cls([cirq.H(c)] * 2)

    c_start = circuit_cls.zip(circuit1, circuit2, circuit3, align='LEFT')
    assert c_start == circuit_cls(
        cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
        cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
        cirq.Moment(cirq.H(a), cirq.H(b)),
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.H(a)),
    )

    c_end = circuit_cls.zip(circuit1, circuit2, circuit3, align='RIGHT')
    assert c_end == circuit_cls(
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.H(a)),
        cirq.Moment(cirq.H(a), cirq.H(b)),
        cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
        cirq.Moment(cirq.H(a), cirq.H(b), cirq.H(c)),
    )


@pytest.mark.parametrize('circuit_cls', [cirq.Circuit, cirq.FrozenCircuit])
def test_repr_html_escaping(circuit_cls):
    class TestGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _circuit_diagram_info_(self, args):
            return cirq.CircuitDiagramInfo(wire_symbols=["< ' F ' >", "< ' F ' >"])

    F2 = TestGate()
    a = cirq.LineQubit(1)
    c = cirq.NamedQubit("|c>")

    circuit = circuit_cls([F2(a, c)])

    # Escaping Special Characters in Gate names.
    assert '&lt; &#x27; F &#x27; &gt;' in circuit._repr_html_()

    # Escaping Special Characters in Qubit names.
    assert '|c&gt;' in circuit._repr_html_()


def test_tetris_concat():
    a, b = cirq.LineQubit.range(2)
    empty = cirq.Circuit()

    assert cirq.Circuit.tetris_concat(empty, empty) == empty
    assert cirq.Circuit.tetris_concat() == empty
    assert empty.tetris_concat(empty) == empty
    assert empty.tetris_concat(empty, empty) == empty

    ha = cirq.Circuit(cirq.H(a))
    hb = cirq.Circuit(cirq.H(b))
    assert ha.tetris_concat(hb) == ha.zip(hb)

    assert ha.tetris_concat(empty) == ha
    assert empty.tetris_concat(ha) == ha

    hac = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b))
    assert hac.tetris_concat(hb) == hac + hb
    assert hb.tetris_concat(hac) == hb.zip(hac)

    zig = cirq.Circuit(cirq.H(a), cirq.CNOT(a, b), cirq.H(b))
    assert zig.tetris_concat(zig) == cirq.Circuit(
        cirq.H(a), cirq.CNOT(a, b), cirq.Moment(cirq.H(a), cirq.H(b)), cirq.CNOT(a, b), cirq.H(b)
    )

    zag = cirq.Circuit(cirq.H(a), cirq.H(a), cirq.CNOT(a, b), cirq.H(b), cirq.H(b))
    assert zag.tetris_concat(zag) == cirq.Circuit(
        cirq.H(a),
        cirq.H(a),
        cirq.CNOT(a, b),
        cirq.Moment(cirq.H(a), cirq.H(b)),
        cirq.Moment(cirq.H(a), cirq.H(b)),
        cirq.CNOT(a, b),
        cirq.H(b),
        cirq.H(b),
    )

    space = cirq.Circuit(cirq.Moment()) * 10
    f = cirq.Circuit.tetris_concat
    assert len(f(space, ha)) == 10
    assert len(f(space, ha, ha, ha)) == 10
    assert len(f(space, f(ha, ha, ha))) == 10
    assert len(f(space, ha, align='LEFT')) == 10
    assert len(f(space, ha, ha, ha, align='RIGHT')) == 12
    assert len(f(space, f(ha, ha, ha, align='LEFT'))) == 10
    assert len(f(space, f(ha, ha, ha, align='RIGHT'))) == 10
    assert len(f(space, f(ha, ha, ha), align='LEFT')) == 10
    assert len(f(space, f(ha, ha, ha), align='RIGHT')) == 10

    # L shape overlap (vary c1).
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 4),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 1),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )
    assert 8 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 6),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )
    assert 9 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 7),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )

    # L shape overlap (vary c2).
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 5, cirq.CZ(a, b)),
        )
    )
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 4, cirq.CZ(a, b)),
        )
    )
    assert 7 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 1, cirq.CZ(a, b)),
        )
    )
    assert 8 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 6, cirq.CZ(a, b)),
        )
    )
    assert 9 == len(
        f(
            cirq.Circuit(cirq.CZ(a, b), [cirq.H(a)] * 5),
            cirq.Circuit([cirq.H(b)] * 7, cirq.CZ(a, b)),
        )
    )

    # When scanning sees a possible hit, continues scanning for earlier hit.
    assert 10 == len(
        f(
            cirq.Circuit(
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(cirq.H(a)),
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(cirq.H(b)),
            ),
            cirq.Circuit(
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(),
                cirq.Moment(cirq.H(a)),
                cirq.Moment(),
                cirq.Moment(cirq.H(b)),
            ),
        )
    )
    # Correct tie breaker when one operation sees two possible hits.
    for cz_order in [cirq.CZ(a, b), cirq.CZ(b, a)]:
        assert 3 == len(
            f(
                cirq.Circuit(
                    cirq.Moment(cz_order),
                    cirq.Moment(),
                    cirq.Moment(),
                ),
                cirq.Circuit(
                    cirq.Moment(cirq.H(a)),
                    cirq.Moment(cirq.H(b)),
                ),
            )
        )

    # Types.
    v = ha.freeze().tetris_concat(empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()
    v = ha.tetris_concat(empty.freeze())
    assert type(v) is cirq.Circuit and v == ha
    v = ha.freeze().tetris_concat(empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()
    v = cirq.Circuit.tetris_concat(ha, empty)
    assert type(v) is cirq.Circuit and v == ha
    v = cirq.FrozenCircuit.tetris_concat(ha, empty)
    assert type(v) is cirq.FrozenCircuit and v == ha.freeze()


def test_tetris_concat_alignment():
    a, b = cirq.LineQubit.range(2)

    assert cirq.Circuit.tetris_concat(
        cirq.Circuit(cirq.X(a)),
        cirq.Circuit(cirq.Y(b)) * 4,
        cirq.Circuit(cirq.Z(a)),
        align='first',
    ) == cirq.Circuit(
        cirq.Moment(cirq.X(a), cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.Z(a), cirq.Y(b)),
    )

    assert cirq.Circuit.tetris_concat(
        cirq.Circuit(cirq.X(a)),
        cirq.Circuit(cirq.Y(b)) * 4,
        cirq.Circuit(cirq.Z(a)),
        align='left',
    ) == cirq.Circuit(
        cirq.Moment(cirq.X(a), cirq.Y(b)),
        cirq.Moment(cirq.Z(a), cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
    )

    assert cirq.Circuit.tetris_concat(
        cirq.Circuit(cirq.X(a)),
        cirq.Circuit(cirq.Y(b)) * 4,
        cirq.Circuit(cirq.Z(a)),
        align='right',
    ) == cirq.Circuit(
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.Y(b)),
        cirq.Moment(cirq.X(a), cirq.Y(b)),
        cirq.Moment(cirq.Z(a)),
    )


def test_factorize_one_factor():
    circuit = cirq.Circuit()
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit.append(
        [cirq.Moment([cirq.CZ(q0, q1), cirq.H(q2)]), cirq.Moment([cirq.H(q0), cirq.CZ(q1, q2)])]
    )
    factors = list(circuit.factorize())
    assert len(factors) == 1
    assert factors[0] == circuit
    desired = """
0: ───@───H───
      │
1: ───@───@───
          │
2: ───H───@───
"""
    cirq.testing.assert_has_diagram(factors[0], desired)


def test_factorize_simple_circuit_two_factors():
    circuit = cirq.Circuit()
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit.append([cirq.H(q1), cirq.CZ(q0, q1), cirq.H(q2), cirq.H(q0), cirq.H(q0)])
    factors = list(circuit.factorize())
    assert len(factors) == 2
    desired = [
        """
0: ───────@───H───H───
          │
1: ───H───@───────────
""",
        """
2: ───H───────────────
""",
    ]
    for f, d in zip(factors, desired):
        cirq.testing.assert_has_diagram(f, d)


def test_factorize_large_circuit():
    circuit = cirq.Circuit()
    qubits = cirq.GridQubit.rect(3, 3)
    circuit.append(cirq.Moment(cirq.X(q) for q in qubits))
    pairset = [[(0, 2), (4, 6)], [(1, 2), (4, 8)]]
    for pairs in pairset:
        circuit.append(cirq.Moment(cirq.CZ(qubits[a], qubits[b]) for (a, b) in pairs))
    circuit.append(cirq.Moment(cirq.Y(q) for q in qubits))
    # expect 5 factors
    factors = list(circuit.factorize())
    desired = [
        """
(0, 0): ───X───@───────Y───
               │
(0, 1): ───X───┼───@───Y───
               │   │
(0, 2): ───X───@───@───Y───
""",
        """
(1, 0): ───X───────────Y───
""",
        """
(1, 1): ───X───@───@───Y───
               │   │
(2, 0): ───X───@───┼───Y───
                   │
(2, 2): ───X───────@───Y───
""",
        """
(1, 2): ───X───────────Y───
""",
        """
(2, 1): ───X───────────Y───
    """,
    ]
    assert len(factors) == 5
    for f, d in zip(factors, desired):
        cirq.testing.assert_has_diagram(f, d)


def test_zero_target_operations_go_below_diagram():
    class CustomOperationAnnotation(cirq.Operation):
        def __init__(self, text: str):
            self.text = text

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        @property
        def qubits(self):
            return ()

        def _circuit_diagram_info_(self, args) -> str:
            return self.text

    class CustomOperationAnnotationNoInfo(cirq.Operation):
        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        @property
        def qubits(self):
            return ()

        def __str__(self):
            return "custom!"

    class CustomGateAnnotation(cirq.Gate):
        def __init__(self, text: str):
            self.text = text

        def _num_qubits_(self):
            return 0

        def _circuit_diagram_info_(self, args) -> str:
            return self.text

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.Moment(
                CustomOperationAnnotation("a"),
                CustomGateAnnotation("b").on(),
                CustomOperationAnnotation("c"),
            ),
            cirq.Moment(
                CustomOperationAnnotation("e"),
                CustomOperationAnnotation("d"),
            ),
        ),
        """
    a   e
    b   d
    c
    """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.Moment(
                cirq.H(cirq.LineQubit(0)),
                CustomOperationAnnotation("a"),
                cirq.GlobalPhaseOperation(1j),
            ),
        ),
        """
0: ─────────────H──────

global phase:   0.5π
                a
    """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.Moment(
                cirq.H(cirq.LineQubit(0)),
                cirq.CircuitOperation(cirq.FrozenCircuit(CustomOperationAnnotation("a"))),
            ),
        ),
        """
0: ───H───
      a
        """,
    )

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.Moment(
                cirq.X(cirq.LineQubit(0)),
                CustomOperationAnnotation("a"),
                CustomGateAnnotation("b").on(),
                CustomOperationAnnotation("c"),
            ),
            cirq.Moment(
                CustomOperationAnnotation("eee"),
                CustomOperationAnnotation("d"),
            ),
            cirq.Moment(
                cirq.CNOT(cirq.LineQubit(0), cirq.LineQubit(2)),
                cirq.CNOT(cirq.LineQubit(1), cirq.LineQubit(3)),
                CustomOperationAnnotationNoInfo(),
                CustomOperationAnnotation("zzz"),
            ),
            cirq.Moment(
                cirq.H(cirq.LineQubit(2)),
            ),
        ),
        """
                ┌────────┐
0: ───X──────────@───────────────
                 │
1: ──────────────┼──────@────────
                 │      │
2: ──────────────X──────┼────H───
                        │
3: ─────────────────────X────────
      a   eee    custom!
      b   d      zzz
      c
                └────────┘
    """,
    )
