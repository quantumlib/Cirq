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

import itertools
import math
from typing import List, cast

import numpy as np
import pytest
import sympy

import cirq
import cirq.testing


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def _sample_qubit_pauli_maps():
    """All combinations of having a Pauli or nothing on 3 qubits.
    Yields 64 qubit pauli maps
    """
    qubits = _make_qubits(3)
    paulis_or_none = (None, cirq.X, cirq.Y, cirq.Z)
    for paulis in itertools.product(paulis_or_none, repeat=len(qubits)):
        yield {qubit: pauli for qubit, pauli in zip(qubits, paulis) if pauli is not None}


def _small_sample_qubit_pauli_maps():
    """A few representative samples of qubit maps.

    Only tests 10 combinations of Paulis to speed up testing.
    """
    qubits = _make_qubits(3)
    yield {}
    yield {qubits[0]: cirq.X}
    yield {qubits[1]: cirq.X}
    yield {qubits[2]: cirq.X}
    yield {qubits[1]: cirq.Z}

    yield {qubits[0]: cirq.Y, qubits[1]: cirq.Z}
    yield {qubits[1]: cirq.Z, qubits[2]: cirq.X}
    yield {qubits[0]: cirq.X, qubits[1]: cirq.X, qubits[2]: cirq.X}
    yield {qubits[0]: cirq.X, qubits[1]: cirq.Y, qubits[2]: cirq.Z}
    yield {qubits[0]: cirq.Z, qubits[1]: cirq.X, qubits[2]: cirq.Y}


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.PauliString(),
        lambda: cirq.PauliString(qubit_pauli_map={}),
        lambda: cirq.PauliString(qubit_pauli_map={}, coefficient=+1),
    )
    eq.add_equality_group(cirq.PauliString(qubit_pauli_map={}, coefficient=-1))
    for q, pauli in itertools.product((q0, q1), (cirq.X, cirq.Y, cirq.Z)):
        eq.add_equality_group(cirq.PauliString(qubit_pauli_map={q: pauli}, coefficient=+1))
        eq.add_equality_group(cirq.PauliString(qubit_pauli_map={q: pauli}, coefficient=-1))
    for q, p0, p1 in itertools.product(
        (q0, q1), (cirq.X, cirq.Y, cirq.Z), (cirq.X, cirq.Y, cirq.Z)
    ):
        eq.add_equality_group(cirq.PauliString(qubit_pauli_map={q: p0, q2: p1}, coefficient=+1))


def test_equal_up_to_coefficient():
    (q0,) = _make_qubits(1)
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, +1))
    assert cirq.PauliString({}, -1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(cirq.PauliString({}, 2j))

    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.X}, +1)
    )
    assert cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.X}, -1)
    )
    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.X}, -1)
    )

    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.Y}, +1)
    )
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.Y}, 1j)
    )
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.Y}, -1)
    )
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
        cirq.PauliString({q0: cirq.Y}, -1)
    )

    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({}, +1))
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(cirq.PauliString({}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(cirq.PauliString({}, -1))


def test_exponentiation_as_exponent():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})

    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        _ = math.e ** (math.pi * p)

    with pytest.raises(TypeError, match='unsupported'):
        _ = 'test' ** p

    assert cirq.approx_eq(
        math.e ** (-0.5j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=-0.5),
    )

    assert cirq.approx_eq(
        math.e ** (0.25j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25),
    )

    assert cirq.approx_eq(
        2 ** (0.25j * math.pi * p),
        cirq.PauliStringPhasor(
            p, exponent_neg=-0.25 * math.log(2), exponent_pos=0.25 * math.log(2)
        ),
    )

    assert cirq.approx_eq(
        np.exp(0.25j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25),
    )


def test_exponentiate_single_value_as_exponent():
    q = cirq.LineQubit(0)

    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.X(q)), cirq.rx(0.25 * math.pi).on(q))

    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.Y(q)), cirq.ry(0.25 * math.pi).on(q))

    assert cirq.approx_eq(math.e ** (-0.125j * math.pi * cirq.Z(q)), cirq.rz(0.25 * math.pi).on(q))

    assert cirq.approx_eq(np.exp(-0.15j * math.pi * cirq.X(q)), cirq.rx(0.3 * math.pi).on(q))

    assert cirq.approx_eq(cirq.X(q) ** 0.5, cirq.XPowGate(exponent=0.5).on(q))

    assert cirq.approx_eq(cirq.Y(q) ** 0.5, cirq.YPowGate(exponent=0.5).on(q))

    assert cirq.approx_eq(cirq.Z(q) ** 0.5, cirq.ZPowGate(exponent=0.5).on(q))


def test_exponentiation_as_base():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})

    with pytest.raises(TypeError, match='unsupported'):
        _ = (2 * p) ** 5

    with pytest.raises(TypeError, match='unsupported'):
        _ = p ** 'test'

    with pytest.raises(TypeError, match='unsupported'):
        _ = p ** 1j

    assert p ** -1 == p

    assert cirq.approx_eq(p ** 0.5, cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=0))

    assert cirq.approx_eq(p ** -0.5, cirq.PauliStringPhasor(p, exponent_neg=-0.5, exponent_pos=0))

    assert cirq.approx_eq(
        math.e ** (0.25j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25),
    )

    assert cirq.approx_eq(
        2 ** (0.25j * math.pi * p),
        cirq.PauliStringPhasor(
            p, exponent_neg=-0.25 * math.log(2), exponent_pos=0.25 * math.log(2)
        ),
    )

    assert cirq.approx_eq(
        np.exp(0.25j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25),
    )

    np.testing.assert_allclose(
        cirq.unitary(np.exp(0.5j * math.pi * cirq.Z(a))),
        np.diag([np.exp(0.5j * math.pi), np.exp(-0.5j * math.pi)]),
        atol=1e-8,
    )


@pytest.mark.parametrize('pauli', (cirq.X, cirq.Y, cirq.Z))
def test_list_op_constructor_matches_mapping(pauli):
    (q0,) = _make_qubits(1)
    op = pauli.on(q0)
    assert cirq.PauliString([op]) == cirq.PauliString({q0: pauli})


def test_constructor_flexibility():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.PauliString(cirq.CZ(a, b))
    with pytest.raises(TypeError, match='cirq.PAULI_STRING_LIKE'):
        _ = cirq.PauliString('test')
    with pytest.raises(TypeError, match='S is not a Pauli'):
        _ = cirq.PauliString(qubit_pauli_map={a: cirq.S})

    assert cirq.PauliString(cirq.X(a)) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([cirq.X(a)]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.X(a)]]]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.I(a)]]]) == cirq.PauliString()

    assert cirq.PauliString(1, 2, 3, cirq.X(a), cirq.Y(a)) == cirq.PauliString(
        qubit_pauli_map={a: cirq.Z}, coefficient=6j
    )

    assert cirq.PauliString(cirq.X(a), cirq.X(a)) == cirq.PauliString()
    assert cirq.PauliString(cirq.X(a), cirq.X(b)) == cirq.PauliString(
        qubit_pauli_map={a: cirq.X, b: cirq.X}
    )

    assert cirq.PauliString(0) == cirq.PauliString(coefficient=0)

    assert cirq.PauliString(1, 2, 3, {a: cirq.X}, cirq.Y(a)) == cirq.PauliString(
        qubit_pauli_map={a: cirq.Z}, coefficient=6j
    )


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_getitem(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = cirq.PauliString(qubit_pauli_map=qubit_pauli_map)
    for key in qubit_pauli_map:
        assert qubit_pauli_map[key] == pauli_string[key]
    with pytest.raises(KeyError):
        _ = qubit_pauli_map[other]
    with pytest.raises(KeyError):
        _ = pauli_string[other]


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_get(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = cirq.PauliString(qubit_pauli_map)
    for key in qubit_pauli_map:
        assert qubit_pauli_map.get(key) == pauli_string.get(key)
    assert qubit_pauli_map.get(other) is None
    assert pauli_string.get(other) is None
    # pylint: disable=too-many-function-args
    assert qubit_pauli_map.get(other, 5) == pauli_string.get(other, 5) == 5
    # pylint: enable=too-many-function-args


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_contains(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = cirq.PauliString(qubit_pauli_map)
    for key in qubit_pauli_map:
        assert key in pauli_string
    assert other not in pauli_string


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_basic_functionality(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    # Test items
    assert len(qubit_pauli_map.items()) == len(pauli_string.items())
    assert set(qubit_pauli_map.items()) == set(pauli_string.items())

    # Test values
    assert len(qubit_pauli_map.values()) == len(pauli_string.values())
    assert set(qubit_pauli_map.values()) == set(pauli_string.values())

    # Test length
    assert len(qubit_pauli_map) == len(pauli_string)

    # Test keys
    assert len(qubit_pauli_map.keys()) == len(pauli_string.keys()) == len(pauli_string.qubits)
    assert set(qubit_pauli_map.keys()) == set(pauli_string.keys()) == set(pauli_string.qubits)

    # Test iteration
    assert len(tuple(qubit_pauli_map)) == len(tuple(pauli_string))
    assert set(tuple(qubit_pauli_map)) == set(tuple(pauli_string))


def test_repr():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = cirq.PauliString({q2: cirq.X, q1: cirq.Y, q0: cirq.Z})
    cirq.testing.assert_equivalent_repr(pauli_string)
    cirq.testing.assert_equivalent_repr(-pauli_string)
    cirq.testing.assert_equivalent_repr(1j * pauli_string)
    cirq.testing.assert_equivalent_repr(2 * pauli_string)
    cirq.testing.assert_equivalent_repr(cirq.PauliString())


def test_str():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = cirq.PauliString({q2: cirq.X, q1: cirq.Y, q0: cirq.Z})
    assert str(cirq.PauliString({})) == 'I'
    assert str(-cirq.PauliString({})) == '-I'
    assert str(pauli_string) == 'Z(q0)*Y(q1)*X(q2)'
    assert str(-pauli_string) == '-Z(q0)*Y(q1)*X(q2)'
    assert str(1j * pauli_string) == '1j*Z(q0)*Y(q1)*X(q2)'
    assert str(pauli_string * -1j) == '-1j*Z(q0)*Y(q1)*X(q2)'


@pytest.mark.parametrize(
    'map1,map2,out',
    (
        lambda q0, q1, q2: (
            ({}, {}, {}),
            ({q0: cirq.X}, {q0: cirq.Y}, {q0: (cirq.X, cirq.Y)}),
            ({q0: cirq.X}, {q1: cirq.X}, {}),
            ({q0: cirq.Y, q1: cirq.Z}, {q1: cirq.Y, q2: cirq.X}, {q1: (cirq.Z, cirq.Y)}),
            ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {}, {}),
            (
                {q0: cirq.X, q1: cirq.Y, q2: cirq.Z},
                {q0: cirq.Y, q1: cirq.Z},
                {q0: (cirq.X, cirq.Y), q1: (cirq.Y, cirq.Z)},
            ),
        )
    )(*_make_qubits(3)),
)
def test_zip_items(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_items(ps2))
    assert len(out_actual) == len(out)
    assert dict(out_actual) == out


@pytest.mark.parametrize(
    'map1,map2,out',
    (
        lambda q0, q1, q2: (
            ({}, {}, ()),
            ({q0: cirq.X}, {q0: cirq.Y}, ((cirq.X, cirq.Y),)),
            ({q0: cirq.X}, {q1: cirq.X}, ()),
            ({q0: cirq.Y, q1: cirq.Z}, {q1: cirq.Y, q2: cirq.X}, ((cirq.Z, cirq.Y),)),
            ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {}, ()),
            (
                {q0: cirq.X, q1: cirq.Y, q2: cirq.Z},
                {q0: cirq.Y, q1: cirq.Z},
                # Order not necessary
                ((cirq.X, cirq.Y), (cirq.Y, cirq.Z)),
            ),
        )
    )(*_make_qubits(3)),
)
def test_zip_paulis(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_paulis(ps2))
    assert len(out_actual) == len(out)
    if len(out) <= 1:
        assert out_actual == out
    assert set(out_actual) == set(out)  # Ignore output order


def test_commutes():
    qubits = _make_qubits(3)

    ps1 = cirq.PauliString([cirq.X(qubits[0])])
    with pytest.raises(TypeError):
        cirq.commutes(ps1, 'X')
    assert cirq.commutes(ps1, 'X', default='default') == 'default'
    for A, commutes in [(cirq.X, True), (cirq.Y, False)]:
        assert cirq.commutes(ps1, cirq.PauliString([A(qubits[0])])) == commutes
        assert cirq.commutes(ps1, cirq.PauliString([A(qubits[1])]))

    ps1 = cirq.PauliString(dict(zip(qubits, (cirq.X, cirq.Y))))

    for paulis, commutes in {
        (cirq.X, cirq.Y): True,
        (cirq.X, cirq.Z): False,
        (cirq.Y, cirq.X): True,
        (cirq.Y, cirq.Z): True,
        (cirq.X, cirq.Y, cirq.Z): True,
        (cirq.X, cirq.Z, cirq.Z): False,
        (cirq.Y, cirq.X, cirq.Z): True,
        (cirq.Y, cirq.Z, cirq.X): True,
    }.items():
        ps2 = cirq.PauliString(dict(zip(qubits, paulis)))
        assert cirq.commutes(ps1, ps2) == commutes

    for paulis, commutes in {
        (cirq.Y, cirq.X): True,
        (cirq.Z, cirq.X): False,
        (cirq.X, cirq.Y): False,
        (cirq.Z, cirq.Y): False,
    }.items():
        ps2 = cirq.PauliString(dict(zip(qubits[1:], paulis)))
        assert cirq.commutes(ps1, ps2) == commutes


def test_negate():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.X, q1: cirq.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    ps2 = cirq.PauliString(qubit_pauli_map, -1)
    assert -ps1 == ps2
    assert ps1 == -ps2
    neg_ps1 = -ps1
    assert -neg_ps1 == ps1

    m = ps1.mutable_copy()
    assert -m == -1 * m
    assert -m is not m
    assert isinstance(-m, cirq.MutablePauliString)


def test_mul_scalar():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert -p == -1 * p == -1.0 * p == p * -1 == p * complex(-1)
    assert -p != 1j * p
    assert +p == 1 * p

    assert p * cirq.I(a) == p
    assert cirq.I(a) * p == p

    with pytest.raises(TypeError, match="sequence by non-int of type 'PauliString'"):
        _ = p * 'test'
    with pytest.raises(TypeError, match="sequence by non-int of type 'PauliString'"):
        _ = 'test' * p


def test_div_scalar():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert -p == p / -1 == p / -1.0 == p / (-1 + 0j)
    assert -p != p / 1j
    assert +p == p / 1
    assert p * 2 == p / 0.5
    with pytest.raises(TypeError):
        _ = p / 'test'
    with pytest.raises(TypeError):
        # noinspection PyUnresolvedReferences
        _ = 'test' / p


def test_mul_strings():
    a, b, c, d = cirq.LineQubit.range(4)
    p1 = cirq.PauliString({a: cirq.X, b: cirq.Y, c: cirq.Z})
    p2 = cirq.PauliString({b: cirq.X, c: cirq.Y, d: cirq.Z})
    assert p1 * p2 == -cirq.PauliString(
        {
            a: cirq.X,
            b: cirq.Z,
            c: cirq.X,
            d: cirq.Z,
        }
    )

    assert cirq.X(a) * cirq.PauliString({a: cirq.X}) == cirq.PauliString()
    assert cirq.PauliString({a: cirq.X}) * cirq.X(a) == cirq.PauliString()
    assert cirq.X(a) * cirq.X(a) == cirq.PauliString()
    assert -cirq.X(a) * -cirq.X(a) == cirq.PauliString()

    with pytest.raises(TypeError, match='unsupported'):
        _ = cirq.X(a) * object()
    with pytest.raises(TypeError, match='unsupported'):
        # noinspection PyUnresolvedReferences
        _ = object() * cirq.X(a)
    assert -cirq.X(a) == -cirq.PauliString({a: cirq.X})


def test_op_equivalence():
    a, b = cirq.LineQubit.range(2)
    various_x = [
        cirq.X(a),
        cirq.PauliString({a: cirq.X}),
        cirq.PauliString([cirq.X.on(a)]),
        cirq.SingleQubitPauliStringGateOperation(cirq.X, a),
        cirq.GateOperation(cirq.X, [a]),
    ]

    for x in various_x:
        cirq.testing.assert_equivalent_repr(x)

    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*various_x)
    eq.add_equality_group(cirq.Y(a), cirq.PauliString({a: cirq.Y}))
    eq.add_equality_group(-cirq.PauliString({a: cirq.X}))
    eq.add_equality_group(cirq.Z(a), cirq.PauliString({a: cirq.Z}))
    eq.add_equality_group(cirq.Z(b), cirq.PauliString({b: cirq.Z}))


def test_op_product():
    a, b = cirq.LineQubit.range(2)

    assert cirq.X(a) * cirq.X(b) == cirq.PauliString({a: cirq.X, b: cirq.X})
    assert cirq.X(a) * cirq.Y(b) == cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert cirq.Z(a) * cirq.Y(b) == cirq.PauliString({a: cirq.Z, b: cirq.Y})

    assert cirq.X(a) * cirq.X(a) == cirq.PauliString()
    assert cirq.X(a) * cirq.Y(a) == 1j * cirq.PauliString({a: cirq.Z})
    assert cirq.Y(a) * cirq.Z(b) * cirq.X(a) == -1j * cirq.PauliString({a: cirq.Z, b: cirq.Z})


def test_pos():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.X, q1: cirq.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    assert ps1 == +ps1

    m = ps1.mutable_copy()
    assert +m == m
    assert +m is not m
    assert isinstance(+m, cirq.MutablePauliString)


def test_pow():
    a, b = cirq.LineQubit.range(2)

    assert cirq.PauliString({a: cirq.X}) ** 0.25 == cirq.X(a) ** 0.25
    assert cirq.PauliString({a: cirq.Y}) ** 0.25 == cirq.Y(a) ** 0.25
    assert cirq.PauliString({a: cirq.Z}) ** 0.25 == cirq.Z(a) ** 0.25

    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert p ** 1 == p
    assert p ** -1 == p
    assert (-p) ** 1 == -p
    assert (-p) ** -1 == -p
    assert (1j * p) ** 1 == 1j * p
    assert (1j * p) ** -1 == -1j * p


def test_rpow():
    a, b = cirq.LineQubit.range(2)

    u = cirq.unitary(np.exp(1j * np.pi / 2 * cirq.Z(a) * cirq.Z(b)))
    np.testing.assert_allclose(u, np.diag([1j, -1j, -1j, 1j]), atol=1e-8)

    u = cirq.unitary(np.exp(-1j * np.pi / 4 * cirq.Z(a) * cirq.Z(b)))
    cirq.testing.assert_allclose_up_to_global_phase(u, np.diag([1, 1j, 1j, 1]), atol=1e-8)

    u = cirq.unitary(np.e ** (1j * np.pi * cirq.Z(a) * cirq.Z(b)))
    np.testing.assert_allclose(u, np.diag([-1, -1, -1, -1]), atol=1e-8)


def test_numpy_ufunc():
    with pytest.raises(TypeError, match="returned NotImplemented"):
        _ = np.sin(cirq.PauliString())
    with pytest.raises(NotImplementedError, match="non-Hermitian"):
        _ = np.exp(cirq.PauliString())
    x = np.exp(1j * np.pi * cirq.PauliString())
    assert x is not None


def test_map_qubits():
    a, b = (cirq.NamedQubit(name) for name in 'ab')
    q0, q1 = _make_qubits(2)
    qubit_pauli_map1 = {a: cirq.X, b: cirq.Y}
    qubit_pauli_map2 = {q0: cirq.X, q1: cirq.Y}
    qubit_map = {a: q0, b: q1}
    ps1 = cirq.PauliString(qubit_pauli_map1)
    ps2 = cirq.PauliString(qubit_pauli_map2)
    assert ps1.map_qubits(qubit_map) == ps2


def test_to_z_basis_ops():
    x0 = np.array([1, 1]) / np.sqrt(2)
    x1 = np.array([1, -1]) / np.sqrt(2)
    y0 = np.array([1, 1j]) / np.sqrt(2)
    y1 = np.array([1, -1j]) / np.sqrt(2)
    z0 = np.array([1, 0])
    z1 = np.array([0, 1])

    q0, q1, q2, q3, q4, q5 = _make_qubits(6)
    pauli_string = cirq.PauliString(
        {q0: cirq.X, q1: cirq.X, q2: cirq.Y, q3: cirq.Y, q4: cirq.Z, q5: cirq.Z}
    )
    circuit = cirq.Circuit(pauli_string.to_z_basis_ops())

    initial_state = cirq.kron(x0, x1, y0, y1, z0, z1, shape_len=1)
    z_basis_state = circuit.final_state_vector(initial_state)

    expected_state = np.zeros(2 ** 6)
    expected_state[0b010101] = 1

    cirq.testing.assert_allclose_up_to_global_phase(
        z_basis_state, expected_state, rtol=1e-7, atol=1e-7
    )


def test_to_z_basis_ops_product_state():
    q0, q1, q2, q3, q4, q5 = _make_qubits(6)
    pauli_string = cirq.PauliString(
        {q0: cirq.X, q1: cirq.X, q2: cirq.Y, q3: cirq.Y, q4: cirq.Z, q5: cirq.Z}
    )
    circuit = cirq.Circuit(pauli_string.to_z_basis_ops())

    initial_state = (
        cirq.KET_PLUS(q0)
        * cirq.KET_MINUS(q1)
        * cirq.KET_IMAG(q2)
        * cirq.KET_MINUS_IMAG(q3)
        * cirq.KET_ZERO(q4)
        * cirq.KET_ONE(q5)
    )
    z_basis_state = circuit.final_state_vector(initial_state)

    expected_state = np.zeros(2 ** 6)
    expected_state[0b010101] = 1

    cirq.testing.assert_allclose_up_to_global_phase(
        z_basis_state, expected_state, rtol=1e-7, atol=1e-7
    )


def _assert_pass_over(ops: List[cirq.Operation], before: cirq.PauliString, after: cirq.PauliString):
    assert before.pass_operations_over(ops[::-1]) == after
    assert after.pass_operations_over(ops, after_to_before=True) == before


@pytest.mark.parametrize('shift,sign', itertools.product(range(3), (-1, +1)))
def test_pass_operations_over_single(shift: int, sign: int):
    q0, q1 = _make_qubits(2)
    X, Y, Z = (
        cirq.Pauli.by_relative_index(cast(cirq.Pauli, pauli), shift)
        for pauli in (cirq.X, cirq.Y, cirq.Z)
    )

    op0 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before: cirq.PauliString[cirq.Qid] = cirq.PauliString({q0: X}, sign)
    ps_after = ps_before
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(X)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = ps_before
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_double_map({Z: (X, False), X: (Z, False)})(q0)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: Z, q1: Y}, sign)
    _assert_pass_over([op0], ps_before, ps_after)

    op1 = cirq.SingleQubitCliffordGate.from_pauli(X)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = -ps_before
    _assert_pass_over([op1], ps_before, ps_after)

    ps_after = cirq.PauliString({q0: Z, q1: Y}, -sign)
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(Z, True)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(X, True)(q0)
    ps_before = cirq.PauliString({q0: X}, sign)
    ps_after = cirq.PauliString({q0: Y}, -sign)
    _assert_pass_over([op0, op1], ps_before, ps_after)


@pytest.mark.parametrize(
    'shift,t_or_f1, t_or_f2,neg', itertools.product(range(3), *((True, False),) * 3)
)
def test_pass_operations_over_double(shift: int, t_or_f1: bool, t_or_f2: bool, neg: bool):
    sign = -1 if neg else +1
    q0, q1, q2 = _make_qubits(3)
    X, Y, Z = (cirq.Pauli.by_relative_index(pauli, shift) for pauli in (cirq.X, cirq.Y, cirq.Z))

    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString(qubit_pauli_map={q0: Z, q2: Y}, coefficient=sign)
    ps_after = cirq.PauliString(qubit_pauli_map={q0: Z, q2: Y}, coefficient=sign)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q2: Y}, sign)
    ps_after = cirq.PauliString({q0: Z, q2: Y, q1: X}, sign)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q1: Y}, sign)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: X, q1: Z}, -1 if neg ^ t_or_f1 ^ t_or_f2 else +1)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(X, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: Y, q1: Z}, +1 if neg ^ t_or_f1 ^ t_or_f2 else -1)
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_operations_over_cz():
    q0, q1 = _make_qubits(2)
    op0 = cirq.CZ(q0, q1)
    ps_before = cirq.PauliString({q0: cirq.Z, q1: cirq.Y})
    ps_after = cirq.PauliString({q1: cirq.Y})
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_operations_over_no_common_qubits():
    class DummyGate(cirq.SingleQubitGate):
        pass

    q0, q1 = _make_qubits(2)
    op0 = DummyGate()(q1)
    ps_before = cirq.PauliString({q0: cirq.Z})
    ps_after = cirq.PauliString({q0: cirq.Z})
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_unsupported_operations_over():
    (q0,) = _make_qubits(1)
    pauli_string = cirq.PauliString({q0: cirq.X})
    with pytest.raises(TypeError, match='not a known Clifford'):
        pauli_string.pass_operations_over([cirq.T(q0)])


def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, -1)
    new_pauli_string = pauli_string.with_qubits(*new_qubits)

    assert new_pauli_string.qubits == tuple(new_qubits)
    for q in new_qubits:
        assert new_pauli_string[q] == cirq.Pauli.by_index(q.x)
    assert new_pauli_string.coefficient == -1


def test_with_coefficient():
    qubits = cirq.LineQubit.range(4)
    qubit_pauli_map = {q: cirq.Pauli.by_index(q.x) for q in qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, 1.23)
    ps2 = pauli_string.with_coefficient(1.0)
    assert ps2.coefficient == 1.0
    assert ps2.equal_up_to_coefficient(pauli_string)
    assert pauli_string != ps2
    assert pauli_string.coefficient == 1.23


@pytest.mark.parametrize('qubit_pauli_map', _small_sample_qubit_pauli_maps())
def test_consistency(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    cirq.testing.assert_implements_consistent_protocols(pauli_string)


def test_scaled_unitary_consistency():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_implements_consistent_protocols(2 * cirq.X(a) * cirq.Y(b))
    cirq.testing.assert_implements_consistent_protocols(1j * cirq.X(a) * cirq.Y(b))


def test_bool():
    a = cirq.LineQubit(0)
    assert not bool(cirq.PauliString({}))
    assert bool(cirq.PauliString({a: cirq.X}))


def _pauli_string_matrix_cases():
    q0, q1, q2 = cirq.LineQubit.range(3)
    return (
        (cirq.X(q0) * 2, None, np.array([[0, 2], [2, 0]])),
        (cirq.X(q0) * cirq.Y(q1), (q0,), np.array([[0, 1], [1, 0]])),
        (cirq.X(q0) * cirq.Y(q1), (q1,), np.array([[0, -1j], [1j, 0]])),
        (
            cirq.X(q0) * cirq.Y(q1),
            None,
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]]),
        ),
        (
            cirq.X(q0) * cirq.Y(q1),
            (q0, q1),
            np.array([[0, 0, 0, -1j], [0, 0, 1j, 0], [0, -1j, 0, 0], [1j, 0, 0, 0]]),
        ),
        (
            cirq.X(q0) * cirq.Y(q1),
            (q1, q0),
            np.array([[0, 0, 0, -1j], [0, 0, -1j, 0], [0, 1j, 0, 0], [1j, 0, 0, 0]]),
        ),
        (cirq.X(q0) * cirq.Y(q1), (q2,), np.eye(2)),
        (
            cirq.X(q0) * cirq.Y(q1),
            (q2, q1),
            np.array([[0, -1j, 0, 0], [1j, 0, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]]),
        ),
        (
            cirq.X(q0) * cirq.Y(q1),
            (q2, q0, q1),
            np.array(
                [
                    [0, 0, 0, -1j, 0, 0, 0, 0],
                    [0, 0, 1j, 0, 0, 0, 0, 0],
                    [0, -1j, 0, 0, 0, 0, 0, 0],
                    [1j, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, -1j],
                    [0, 0, 0, 0, 0, 0, 1j, 0],
                    [0, 0, 0, 0, 0, -1j, 0, 0],
                    [0, 0, 0, 0, 1j, 0, 0, 0],
                ]
            ),
        ),
    )


@pytest.mark.parametrize('pauli_string, qubits, expected_matrix', _pauli_string_matrix_cases())
def test_matrix(pauli_string, qubits, expected_matrix):
    assert np.allclose(pauli_string.matrix(qubits), expected_matrix)


def test_unitary_matrix():
    a, b = cirq.LineQubit.range(2)
    assert not cirq.has_unitary(2 * cirq.X(a) * cirq.Z(b))
    assert cirq.unitary(2 * cirq.X(a) * cirq.Z(b), default=None) is None
    np.testing.assert_allclose(
        cirq.unitary(cirq.X(a) * cirq.Z(b)),
        np.array(
            [
                [0, 0, 1, 0],
                [0, 0, 0, -1],
                [1, 0, 0, 0],
                [0, -1, 0, 0],
            ]
        ),
    )
    np.testing.assert_allclose(
        cirq.unitary(1j * cirq.X(a) * cirq.Z(b)),
        np.array(
            [
                [0, 0, 1j, 0],
                [0, 0, 0, -1j],
                [1j, 0, 0, 0],
                [0, -1j, 0, 0],
            ]
        ),
    )


def test_decompose():
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose_once(2 * cirq.X(a) * cirq.Z(b), default=None) is None
    assert cirq.decompose_once(1j * cirq.X(a) * cirq.Z(b)) == [
        cirq.GlobalPhaseOperation(1j),
        cirq.X(a),
        cirq.Z(b),
    ]
    assert cirq.decompose_once(cirq.Y(b) * cirq.Z(a)) == [cirq.Z(a), cirq.Y(b)]


def test_rejects_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.PauliString({q: cirq.S})


def test_cannot_multiply_by_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.X(q) * cirq.Z(q) ** 0.5
    with pytest.raises(TypeError):
        _ = cirq.Z(q) ** 0.5 * cirq.X(q)
    with pytest.raises(TypeError):
        _ = cirq.Y(q) * cirq.S(q)


def test_filters_identities():
    q1, q2 = cirq.LineQubit.range(2)
    assert cirq.PauliString({q1: cirq.I, q2: cirq.X}) == cirq.PauliString({q2: cirq.X})


def test_expectation_from_state_vector_invalid_input():
    q0, q1, q2, q3 = _make_qubits(4)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    wf = np.array([1, 0, 0, 0], dtype=np.complex64)
    q_map = {q0: 0, q1: 1}

    im_ps = (1j + 1) * ps
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_ps.expectation_from_state_vector(wf, q_map)

    with pytest.raises(TypeError, match='dtype'):
        ps.expectation_from_state_vector(np.array([1, 0], dtype=np.int), q_map)

    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_state_vector(wf, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_state_vector(wf, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_state_vector(wf, {q0: "bad value"})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_state_vector(wf, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_state_vector(wf, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_state_vector(wf, {q0: -1, q1: 1})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_state_vector(wf, {q0: 0, q1: 3})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_state_vector(wf, {q0: 0, q1: 0})
    # Excess keys are ignored.
    _ = ps.expectation_from_state_vector(wf, {q0: 0, q1: 1, q2: 0})

    # Incorrectly shaped state_vector input.
    with pytest.raises(ValueError, match='7'):
        ps.expectation_from_state_vector(np.arange(7, dtype=np.complex64), q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='normalized'):
        ps.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map_2)

    # The ambiguous case: Density matrices satisfying L2 normalization.
    rho_or_wf = 0.5 * np.ones((2, 2), dtype=np.complex64)
    _ = ps.expectation_from_state_vector(rho_or_wf, q_map)

    wf = np.arange(16, dtype=np.complex64) / np.linalg.norm(np.arange(16))
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_state_vector(wf.reshape((16, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_state_vector(wf.reshape((4, 4, 1)), q_map_2)


def test_expectation_from_state_vector_check_preconditions():
    q0, q1, q2, q3 = _make_qubits(4)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    q_map = {q0: 0, q1: 1, q2: 2, q3: 3}

    with pytest.raises(ValueError, match='normalized'):
        ps.expectation_from_state_vector(np.arange(16, dtype=np.complex64), q_map)

    _ = ps.expectation_from_state_vector(
        np.arange(16, dtype=np.complex64), q_map, check_preconditions=False
    )


def test_expectation_from_state_vector_basis_states():
    q0 = cirq.LineQubit(0)
    x0 = cirq.PauliString({q0: cirq.X})
    q_map = {q0: 0}

    np.testing.assert_allclose(
        x0.expectation_from_state_vector(np.array([1, 0], dtype=np.complex), q_map), 0, atol=1e-7
    )
    np.testing.assert_allclose(
        x0.expectation_from_state_vector(np.array([0, 1], dtype=np.complex), q_map), 0, atol=1e-7
    )
    np.testing.assert_allclose(
        x0.expectation_from_state_vector(np.array([1, 1], dtype=np.complex) / np.sqrt(2), q_map),
        1,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        x0.expectation_from_state_vector(np.array([1, -1], dtype=np.complex) / np.sqrt(2), q_map),
        -1,
        atol=1e-7,
    )

    y0 = cirq.PauliString({q0: cirq.Y})
    np.testing.assert_allclose(
        y0.expectation_from_state_vector(np.array([1, 1j], dtype=np.complex) / np.sqrt(2), q_map),
        1,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        y0.expectation_from_state_vector(np.array([1, -1j], dtype=np.complex) / np.sqrt(2), q_map),
        -1,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        y0.expectation_from_state_vector(np.array([1, 1], dtype=np.complex) / np.sqrt(2), q_map),
        0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        y0.expectation_from_state_vector(np.array([1, -1], dtype=np.complex) / np.sqrt(2), q_map),
        0,
        atol=1e-7,
    )


def test_expectation_from_state_vector_entangled_states():
    q0, q1 = _make_qubits(2)
    z0z1_pauli_map = {q0: cirq.Z, q1: cirq.Z}
    z0z1 = cirq.PauliString(z0z1_pauli_map)
    x0x1_pauli_map = {q0: cirq.X, q1: cirq.X}
    x0x1 = cirq.PauliString(x0x1_pauli_map)
    q_map = {q0: 0, q1: 1}
    wf1 = np.array([0, 1, 1, 0], dtype=np.complex) / np.sqrt(2)
    for state in [wf1, wf1.reshape(2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), -1)
        np.testing.assert_allclose(x0x1.expectation_from_state_vector(state, q_map), 1)

    wf2 = np.array([1, 0, 0, 1], dtype=np.complex) / np.sqrt(2)
    for state in [wf2, wf2.reshape(2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), 1)
        np.testing.assert_allclose(x0x1.expectation_from_state_vector(state, q_map), 1)

    wf3 = np.array([1, 1, 1, 1], dtype=np.complex) / 2
    for state in [wf3, wf3.reshape(2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(x0x1.expectation_from_state_vector(state, q_map), 1)


def test_expectation_from_state_vector_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.complex) / np.sqrt(2)
    for state in [wf, wf.reshape(2, 2, 2)]:
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 0, q1: 1, q2: 2}), 1)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 0, q1: 2, q2: 1}), 1)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 1, q1: 0, q2: 2}), 0)
        np.testing.assert_allclose(z.expectation_from_state_vector(state, {q0: 1, q1: 2, q2: 0}), 0)
        np.testing.assert_allclose(
            z.expectation_from_state_vector(state, {q0: 2, q1: 0, q2: 1}), -1
        )
        np.testing.assert_allclose(
            z.expectation_from_state_vector(state, {q0: 2, q1: 1, q2: 0}), -1
        )


def test_pauli_string_expectation_from_state_vector_pure_state():
    qubits = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.H(qubits[2]),
        cirq.X(qubits[3]),
        cirq.H(qubits[3]),
    )
    wf = circuit.final_state_vector(qubit_order=qubits)

    z0z1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
    z0z2 = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})
    z0z3 = cirq.PauliString({qubits[0]: cirq.Z, qubits[3]: cirq.Z})
    z0x1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.X})
    z1x2 = cirq.PauliString({qubits[1]: cirq.Z, qubits[2]: cirq.X})
    x0z1 = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    x3 = cirq.PauliString({qubits[3]: cirq.X})

    for state in [wf, wf.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), -1)
        np.testing.assert_allclose(z0z2.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(z0z3.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(z0x1.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(z1x2.expectation_from_state_vector(state, q_map), -1)
        np.testing.assert_allclose(x0z1.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(x3.expectation_from_state_vector(state, q_map), -1)


def test_pauli_string_expectation_from_state_vector_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qs)}

    circuit = cirq.Circuit(
        cirq.X(qs[1]),
        cirq.H(qs[2]),
        cirq.X(qs[3]),
        cirq.H(qs[3]),
    )
    wf = circuit.final_state_vector(qubit_order=qs)

    z0z1 = cirq.Z(qs[0]) * cirq.Z(qs[1]) * 0.123
    z0z2 = cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1
    z1x2 = -cirq.Z(qs[1]) * cirq.X(qs[2])

    for state in [wf, wf.reshape((2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_state_vector(state, q_map), -0.123)
        np.testing.assert_allclose(z0z2.expectation_from_state_vector(state, q_map), 0)
        np.testing.assert_allclose(z1x2.expectation_from_state_vector(state, q_map), 1)


def test_expectation_from_density_matrix_invalid_input():
    q0, q1, q2, q3 = _make_qubits(4)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    wf = cirq.testing.random_superposition(4)
    rho = np.kron(wf.conjugate().T, wf).reshape(4, 4)
    q_map = {q0: 0, q1: 1}

    im_ps = (1j + 1) * ps
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_ps.expectation_from_density_matrix(rho, q_map)

    with pytest.raises(TypeError, match='dtype'):
        ps.expectation_from_density_matrix(0.5 * np.eye(2, dtype=np.int), q_map)

    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_density_matrix(rho, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_density_matrix(rho, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
        # noinspection PyTypeChecker
        ps.expectation_from_density_matrix(rho, {q0: "bad value"})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_density_matrix(rho, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_density_matrix(rho, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_density_matrix(rho, {q0: -1, q1: 1})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_density_matrix(rho, {q0: 0, q1: 3})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_density_matrix(rho, {q0: 0, q1: 0})
    # Excess keys are ignored.
    _ = ps.expectation_from_density_matrix(rho, {q0: 0, q1: 1, q2: 0})

    with pytest.raises(ValueError, match='hermitian'):
        ps.expectation_from_density_matrix(1j * np.eye(4), q_map)
    with pytest.raises(ValueError, match='trace'):
        ps.expectation_from_density_matrix(np.eye(4, dtype=np.complex64), q_map)
    with pytest.raises(ValueError, match='semidefinite'):
        ps.expectation_from_density_matrix(
            np.array(
                [[1.1, 0, 0, 0], [0, -0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex64
            ),
            q_map,
        )

    # Incorrectly shaped density matrix input.
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(np.ones((4, 5), dtype=np.complex64), q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(rho.reshape((4, 4, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(rho.reshape((-1)), q_map_2)

    # Correctly shaped state_vectors.
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(np.array([1, 0], dtype=np.complex64), q_map)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(wf, q_map)

    # The ambiguous cases: state_vectors satisfying trace normalization.
    # This also throws an unrelated warning, which is a bug. See #2041.
    rho_or_wf = 0.25 * np.ones((4, 4), dtype=np.complex64)
    _ = ps.expectation_from_density_matrix(rho_or_wf, q_map)


def test_expectation_from_density_matrix_check_preconditions():
    q0, q1 = _make_qubits(2)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    q_map = {q0: 0, q1: 1}

    with pytest.raises(ValueError, match='semidefinite'):
        ps.expectation_from_density_matrix(
            np.array(
                [[1.1, 0, 0, 0], [0, -0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex64
            ),
            q_map,
        )

    _ = ps.expectation_from_density_matrix(
        np.array([[1.1, 0, 0, 0], [0, -0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.complex64),
        q_map,
        check_preconditions=False,
    )


def test_expectation_from_density_matrix_basis_states():
    q0 = cirq.LineQubit(0)
    x0_pauli_map = {q0: cirq.X}
    x0 = cirq.PauliString(x0_pauli_map)
    q_map = {q0: 0}
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(np.array([[1, 0], [0, 0]], dtype=np.complex), q_map), 0
    )
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(np.array([[0, 0], [0, 1]], dtype=np.complex), q_map), 0
    )
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(np.array([[1, 1], [1, 1]], dtype=np.complex) / 2, q_map),
        1,
    )
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(
            np.array([[1, -1], [-1, 1]], dtype=np.complex) / 2, q_map
        ),
        -1,
    )


def test_expectation_from_density_matrix_entangled_states():
    q0, q1 = _make_qubits(2)
    z0z1_pauli_map = {q0: cirq.Z, q1: cirq.Z}
    z0z1 = cirq.PauliString(z0z1_pauli_map)
    x0x1_pauli_map = {q0: cirq.X, q1: cirq.X}
    x0x1 = cirq.PauliString(x0x1_pauli_map)
    q_map = {q0: 0, q1: 1}

    wf1 = np.array([0, 1, 1, 0], dtype=np.complex) / np.sqrt(2)
    rho1 = np.kron(wf1, wf1).reshape(4, 4)
    for state in [rho1, rho1.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)

    wf2 = np.array([1, 0, 0, 1], dtype=np.complex) / np.sqrt(2)
    rho2 = np.kron(wf2, wf2).reshape(4, 4)
    for state in [rho2, rho2.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), 1)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)

    wf3 = np.array([1, 1, 1, 1], dtype=np.complex) / 2
    rho3 = np.kron(wf3, wf3).reshape(4, 4)
    for state in [rho3, rho3.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(x0x1.expectation_from_density_matrix(state, q_map), 1)


def test_expectation_from_density_matrix_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.complex) / np.sqrt(2)
    rho = np.kron(wf, wf).reshape(8, 8)

    for state in [rho, rho.reshape(2, 2, 2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 0, q1: 1, q2: 2}), 1
        )
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 0, q1: 2, q2: 1}), 1
        )
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 1, q1: 0, q2: 2}), 0
        )
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 1, q1: 2, q2: 0}), 0
        )
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 2, q1: 0, q2: 1}), -1
        )
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {q0: 2, q1: 1, q2: 0}), -1
        )


def test_pauli_string_expectation_from_density_matrix_pure_state():
    qubits = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.H(qubits[2]),
        cirq.X(qubits[3]),
        cirq.H(qubits[3]),
    )
    state_vector = circuit.final_state_vector(qubit_order=qubits)
    rho = np.outer(state_vector, np.conj(state_vector))

    z0z1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
    z0z2 = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})
    z0z3 = cirq.PauliString({qubits[0]: cirq.Z, qubits[3]: cirq.Z})
    z0x1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.X})
    z1x2 = cirq.PauliString({qubits[1]: cirq.Z, qubits[2]: cirq.X})
    x0z1 = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    x3 = cirq.PauliString({qubits[3]: cirq.X})

    for state in [rho, rho.reshape((2, 2, 2, 2, 2, 2, 2, 2))]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(z0z2.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(z0z3.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(z0x1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(z1x2.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(x0z1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(x3.expectation_from_density_matrix(state, q_map), -1)


def test_pauli_string_expectation_from_density_matrix_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qs)}

    circuit = cirq.Circuit(
        cirq.X(qs[1]),
        cirq.H(qs[2]),
        cirq.X(qs[3]),
        cirq.H(qs[3]),
    )
    state_vector = circuit.final_state_vector(qubit_order=qs)
    rho = np.outer(state_vector, np.conj(state_vector))

    z0z1 = cirq.Z(qs[0]) * cirq.Z(qs[1]) * 0.123
    z0z2 = cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1
    z1x2 = -cirq.Z(qs[1]) * cirq.X(qs[2])

    for state in [rho, rho.reshape(2, 2, 2, 2, 2, 2, 2, 2)]:
        np.testing.assert_allclose(z0z1.expectation_from_density_matrix(state, q_map), -0.123)
        np.testing.assert_allclose(z0z2.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(z1x2.expectation_from_density_matrix(state, q_map), 1)


def test_pauli_string_expectation_from_state_vector_mixed_state_linearity():
    n_qubits = 6

    state_vector1 = cirq.testing.random_superposition(2 ** n_qubits)
    state_vector2 = cirq.testing.random_superposition(2 ** n_qubits)
    rho1 = np.outer(state_vector1, np.conj(state_vector1))
    rho2 = np.outer(state_vector2, np.conj(state_vector2))
    density_matrix = rho1 / 2 + rho2 / 2

    qubits = cirq.LineQubit.range(n_qubits)
    q_map = {q: i for i, q in enumerate(qubits)}
    paulis = [cirq.X, cirq.Y, cirq.Z]
    pauli_string = cirq.PauliString({q: np.random.choice(paulis) for q in qubits})

    a = pauli_string.expectation_from_state_vector(state_vector1, q_map)
    b = pauli_string.expectation_from_state_vector(state_vector2, q_map)
    c = pauli_string.expectation_from_density_matrix(density_matrix, q_map)
    np.testing.assert_allclose(0.5 * (a + b), c)


def test_conjugated_by_normal_gates():
    a = cirq.LineQubit(0)

    assert cirq.X(a).conjugated_by(cirq.H(a)) == cirq.Z(a)
    assert cirq.Y(a).conjugated_by(cirq.H(a)) == -cirq.Y(a)
    assert cirq.Z(a).conjugated_by(cirq.H(a)) == cirq.X(a)

    assert cirq.X(a).conjugated_by(cirq.S(a)) == -cirq.Y(a)
    assert cirq.Y(a).conjugated_by(cirq.S(a)) == cirq.X(a)
    assert cirq.Z(a).conjugated_by(cirq.S(a)) == cirq.Z(a)


def test_dense():
    a, b, c, d, e = cirq.LineQubit.range(5)
    p = cirq.PauliString([cirq.X(a), cirq.Y(b), cirq.Z(c)])
    assert p.dense([a, b, c, d]) == cirq.DensePauliString('XYZI')
    assert p.dense([d, e, a, b, c]) == cirq.DensePauliString('IIXYZ')
    assert -p.dense([a, b, c, d]) == -cirq.DensePauliString('XYZI')

    with pytest.raises(ValueError, match=r'not self.keys\(\) <= set\(qubits\)'):
        _ = p.dense([a, b])
    with pytest.raises(ValueError, match=r'not self.keys\(\) <= set\(qubits\)'):
        _ = p.dense([a, b, d])


def test_conjugated_by_incorrectly_powered_cliffords():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString([cirq.X(a), cirq.Z(b)])
    cliffords = [
        cirq.H(a),
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.H(a),
        cirq.CNOT(a, b),
        cirq.CZ(a, b),
        cirq.SWAP(a, b),
        cirq.ISWAP(a, b),
        cirq.XX(a, b),
        cirq.YY(a, b),
        cirq.ZZ(a, b),
    ]
    for c in cliffords:
        with pytest.raises(TypeError, match='not a known Clifford'):
            _ = p.conjugated_by(c ** 0.1)
        with pytest.raises(TypeError, match='not a known Clifford'):
            _ = p.conjugated_by(c ** sympy.Symbol('t'))


def test_conjugated_by_global_phase():
    a = cirq.LineQubit(0)
    assert cirq.X(a).conjugated_by(cirq.GlobalPhaseOperation(1j)) == cirq.X(a)
    assert cirq.Z(a).conjugated_by(cirq.GlobalPhaseOperation(np.exp(1.1j))) == cirq.Z(a)

    class DecomposeGlobal(cirq.Gate):
        def num_qubits(self):
            return 1

        def _decompose_(self, qubits):
            yield cirq.GlobalPhaseOperation(1j)

    assert cirq.X(a).conjugated_by(DecomposeGlobal().on(a)) == cirq.X(a)


def test_conjugated_by_composite_with_disjoint_sub_gates():
    a, b = cirq.LineQubit.range(2)

    class DecomposeDisjoint(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            yield cirq.H(qubits[1])

    assert cirq.X(a).conjugated_by(DecomposeDisjoint().on(a, b)) == cirq.X(a)
    assert cirq.X(a).pass_operations_over([DecomposeDisjoint().on(a, b)]) == cirq.X(a)


def test_conjugated_by_clifford_composite():
    class UnknownGate(cirq.Gate):
        def num_qubits(self) -> int:
            return 4

        def _decompose_(self, qubits):
            # Involved.
            yield cirq.SWAP(qubits[0], qubits[1])
            # Uninvolved.
            yield cirq.SWAP(qubits[2], qubits[3])

    a, b, c, d = cirq.LineQubit.range(4)
    p = cirq.X(a) * cirq.Z(b)
    u = UnknownGate()
    assert p.conjugated_by(u(a, b, c, d)) == cirq.Z(a) * cirq.X(b)


def test_conjugated_by_move_into_uninvolved():
    a, b, c, d = cirq.LineQubit.range(4)
    p = cirq.X(a) * cirq.Z(b)
    assert (
        p.conjugated_by(
            [
                cirq.SWAP(c, d),
                cirq.SWAP(b, c),
            ]
        )
        == cirq.X(a) * cirq.Z(d)
    )
    assert (
        p.conjugated_by(
            [
                cirq.SWAP(b, c),
                cirq.SWAP(c, d),
            ]
        )
        == cirq.X(a) * cirq.Z(c)
    )


def test_conjugated_by_common_single_qubit_gates():
    a, b = cirq.LineQubit.range(2)

    base_single_qubit_gates = [
        cirq.I,
        cirq.X,
        cirq.Y,
        cirq.Z,
        cirq.X ** -0.5,
        cirq.Y ** -0.5,
        cirq.Z ** -0.5,
        cirq.X ** 0.5,
        cirq.Y ** 0.5,
        cirq.Z ** 0.5,
        cirq.H,
    ]
    single_qubit_gates = [g ** i for i in range(4) for g in base_single_qubit_gates]
    for p in [cirq.X, cirq.Y, cirq.Z]:
        for g in single_qubit_gates:
            assert p.on(a).conjugated_by(g.on(b)) == p.on(a)

            actual = cirq.unitary(p.on(a).conjugated_by(g.on(a)))
            u = cirq.unitary(g)
            expected = np.conj(u.T) @ cirq.unitary(p) @ u
            assert cirq.allclose_up_to_global_phase(actual, expected, atol=1e-8)


def test_conjugated_by_common_two_qubit_gates():
    class OrderSensitiveGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** -0.5, cirq.CNOT(*qubits)]

    a, b, c, d = cirq.LineQubit.range(4)
    two_qubit_gates = [
        cirq.CNOT,
        cirq.CZ,
        cirq.ISWAP,
        cirq.ISWAP ** -1,
        cirq.SWAP,
        cirq.XX ** 0.5,
        cirq.YY ** 0.5,
        cirq.ZZ ** 0.5,
        cirq.XX,
        cirq.YY,
        cirq.ZZ,
        cirq.XX ** -0.5,
        cirq.YY ** -0.5,
        cirq.ZZ ** -0.5,
    ]
    two_qubit_gates.extend(
        [
            OrderSensitiveGate(),
        ]
    )
    for p1 in [cirq.I, cirq.X, cirq.Y, cirq.Z]:
        for p2 in [cirq.I, cirq.X, cirq.Y, cirq.Z]:
            pd = cirq.DensePauliString([p1, p2])
            p = pd.sparse()
            for g in two_qubit_gates:
                assert p.conjugated_by(g.on(c, d)) == p

                actual = cirq.unitary(p.conjugated_by(g.on(a, b)).dense([a, b]))
                u = cirq.unitary(g)
                expected = np.conj(u.T) @ cirq.unitary(pd) @ u
                np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_conjugated_by_ordering():
    class OrderSensitiveGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** -0.5, cirq.CNOT(*qubits)]

    a, b = cirq.LineQubit.range(2)
    inp = cirq.Z(b)
    out1 = inp.conjugated_by(OrderSensitiveGate().on(a, b))
    out2 = inp.conjugated_by([cirq.H(a), cirq.CNOT(a, b)])
    out3 = inp.conjugated_by(cirq.CNOT(a, b)).conjugated_by(cirq.H(a))
    assert out1 == out2 == out3 == cirq.X(a) * cirq.Z(b)


def test_pass_operations_over_ordering():
    class OrderSensitiveGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** -0.5, cirq.CNOT(*qubits)]

    a, b = cirq.LineQubit.range(2)
    inp = cirq.Z(b)
    out1 = inp.pass_operations_over([OrderSensitiveGate().on(a, b)])
    out2 = inp.pass_operations_over([cirq.CNOT(a, b), cirq.Y(a) ** -0.5])
    out3 = inp.pass_operations_over([cirq.CNOT(a, b)]).pass_operations_over([cirq.Y(a) ** -0.5])
    assert out1 == out2 == out3 == cirq.X(a) * cirq.Z(b)


def test_pass_operations_over_ordering_reversed():
    class OrderSensitiveGate(cirq.Gate):
        def num_qubits(self):
            return 2

        def _decompose_(self, qubits):
            return [cirq.Y(qubits[0]) ** -0.5, cirq.CNOT(*qubits)]

    a, b = cirq.LineQubit.range(2)
    inp = cirq.X(a) * cirq.Z(b)
    out1 = inp.pass_operations_over([OrderSensitiveGate().on(a, b)], after_to_before=True)
    out2 = inp.pass_operations_over([cirq.Y(a) ** -0.5, cirq.CNOT(a, b)], after_to_before=True)
    out3 = inp.pass_operations_over([cirq.Y(a) ** -0.5], after_to_before=True).pass_operations_over(
        [cirq.CNOT(a, b)], after_to_before=True
    )
    assert out1 == out2 == out3 == cirq.Z(b)


def test_pretty_print():
    a, b, c = cirq.LineQubit.range(3)
    result = cirq.PauliString({a: 'x', b: 'y', c: 'z'})

    # Test Jupyter console output from
    class FakePrinter:
        def __init__(self):
            self.text_pretty = ''

        def text(self, to_print):
            self.text_pretty += to_print

    p = FakePrinter()
    result._repr_pretty_(p, False)
    assert p.text_pretty == 'X(0)*Y(1)*Z(2)'

    # Test cycle handling
    p = FakePrinter()
    result._repr_pretty_(p, True)
    assert p.text_pretty == 'cirq.PauliString(...)'


def test_deprecated():
    a = cirq.LineQubit(0)
    state_vector = np.array([1, 1], dtype=np.complex64) / np.sqrt(2)
    with cirq.testing.assert_logs(
        'expectation_from_wavefunction', 'expectation_from_state_vector', 'deprecated'
    ):
        _ = cirq.PauliString({a: 'x'}).expectation_from_wavefunction(state_vector, {a: 0})

    with cirq.testing.assert_logs('state', 'state_vector', 'deprecated'):
        # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        _ = cirq.PauliString({a: 'x'}).expectation_from_state_vector(
            state=state_vector, qubit_map={a: 0}
        )


# pylint: disable=line-too-long
def test_circuit_diagram_info():
    a, b, c = cirq.LineQubit.range(3)

    assert cirq.circuit_diagram_info(cirq.PauliString(), default=None) is None

    cirq.testing.assert_has_diagram(
        cirq.Circuit(
            cirq.PauliString({a: cirq.X}),
            -cirq.PauliString({a: cirq.X}),
            cirq.X(a) * cirq.Z(c),
            1j * cirq.X(a) * cirq.Y(b),
            -1j * cirq.Y(b),
            1j ** 0.5 * cirq.X(a) * cirq.Y(b),
        ),
        """
0: ───PauliString(+X)───PauliString(-X)───PauliString(+X)───PauliString(iX)──────────────────────PauliString((0.707+0.707i)*X)───
                                          │                 │                                    │
1: ───────────────────────────────────────┼─────────────────Y─────────────────PauliString(-iY)───Y───────────────────────────────
                                          │
2: ───────────────────────────────────────Z──────────────────────────────────────────────────────────────────────────────────────
        """,
    )


# pylint: enable=line-too-long


def test_mutable_pauli_string_equality():
    eq = cirq.testing.EqualsTester()
    a, b, c = cirq.LineQubit.range(3)

    eq.add_equality_group(
        cirq.MutablePauliString(),
        cirq.MutablePauliString(),
        cirq.MutablePauliString(1),
        cirq.MutablePauliString(-1, -1),
        cirq.MutablePauliString({a: 0}),
        cirq.MutablePauliString({a: "I"}),
        cirq.MutablePauliString({a: cirq.I}),
        cirq.MutablePauliString(cirq.I(a)),
        cirq.MutablePauliString(cirq.I(b)),
    )

    eq.add_equality_group(
        cirq.MutablePauliString({a: "X"}),
        cirq.MutablePauliString({a: 1}),
        cirq.MutablePauliString({a: cirq.X}),
        cirq.MutablePauliString(cirq.X(a)),
    )

    eq.add_equality_group(
        cirq.MutablePauliString({b: "X"}),
        cirq.MutablePauliString({b: 1}),
        cirq.MutablePauliString({b: cirq.X}),
        cirq.MutablePauliString(cirq.X(b)),
        cirq.MutablePauliString(-1j, cirq.Y(b), cirq.Z(b)),
    )

    eq.add_equality_group(
        cirq.MutablePauliString({a: "X", b: "Y", c: "Z"}),
        cirq.MutablePauliString({a: 1, b: 2, c: 3}),
        cirq.MutablePauliString({a: cirq.X, b: cirq.Y, c: cirq.Z}),
        cirq.MutablePauliString(cirq.X(a) * cirq.Y(b) * cirq.Z(c)),
        cirq.MutablePauliString(cirq.MutablePauliString(cirq.X(a) * cirq.Y(b) * cirq.Z(c))),
        cirq.MutablePauliString(cirq.MutablePauliString(cirq.X(a), cirq.Y(b), cirq.Z(c))),
    )

    # Cross-type equality. (Can't use tester because hashability differs.)
    p = cirq.X(a) * cirq.Y(b)
    assert p == cirq.MutablePauliString(p)

    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        _ = cirq.MutablePauliString("test")
    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        # noinspection PyTypeChecker
        _ = cirq.MutablePauliString(object())


def test_mutable_pauli_string_inplace_multiplication():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString()
    original = p

    # Support for *=.
    p *= cirq.X(a)
    assert p == cirq.X(a) and p is original

    # Bad operand.
    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        p.inplace_left_multiply_by([cirq.X(a), cirq.CZ(a, b), cirq.Z(b)])
    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        p.inplace_left_multiply_by(cirq.CZ(a, b))
    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        p.inplace_right_multiply_by([cirq.X(a), cirq.CZ(a, b), cirq.Z(b)])
    with pytest.raises(TypeError, match="cirq.PAULI_STRING_LIKE"):
        p.inplace_right_multiply_by(cirq.CZ(a, b))
    assert p == cirq.X(a) and p is original

    # Correct order of *=.
    p *= cirq.Y(a)
    assert p == -1j * cirq.Z(a) and p is original
    p *= cirq.Y(a)
    assert p == cirq.X(a) and p is original

    # Correct order of inplace_left_multiply_by.
    p.inplace_left_multiply_by(cirq.Y(a))
    assert p == 1j * cirq.Z(a) and p is original
    p.inplace_left_multiply_by(cirq.Y(a))
    assert p == cirq.X(a) and p is original

    # Correct order of inplace_right_multiply_by.
    p.inplace_right_multiply_by(cirq.Y(a))
    assert p == -1j * cirq.Z(a) and p is original
    p.inplace_right_multiply_by(cirq.Y(a))
    assert p == cirq.X(a) and p is original

    # Multi-qubit case.
    p *= -1 * cirq.X(a) * cirq.X(b)
    assert p == -cirq.X(b) and p is original

    # Support for PAULI_STRING_LIKE
    p.inplace_left_multiply_by({c: 'Z'})
    assert p == -cirq.X(b) * cirq.Z(c) and p is original
    p.inplace_right_multiply_by({c: 'Z'})
    assert p == -cirq.X(b) and p is original


def test_mutable_frozen_copy():
    a, b, c = cirq.LineQubit.range(3)
    p = -cirq.X(a) * cirq.Y(b) * cirq.Z(c)

    pf = p.frozen()
    pm = p.mutable_copy()
    pmm = pm.mutable_copy()
    pmf = pm.frozen()

    assert isinstance(p, cirq.PauliString)
    assert isinstance(pf, cirq.PauliString)
    assert isinstance(pm, cirq.MutablePauliString)
    assert isinstance(pmm, cirq.MutablePauliString)
    assert isinstance(pmf, cirq.PauliString)

    assert p is pf
    assert pm is not pmm
    assert p == pf == pm == pmm == pmf


def test_mutable_pauli_string_inplace_conjugate_by():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString(cirq.X(a))

    class NoOp(cirq.Operation):
        def __init__(self, *qubits):
            self._qubits = qubits

        @property
        def qubits(self):
            # coverage: ignore
            return self._qubits

        def with_qubits(self, *new_qubits):
            raise NotImplementedError()

        def _decompose_(self):
            return []

    # No-ops
    p2 = p.inplace_after(cirq.GlobalPhaseOperation(1j))
    assert p2 is p and p == cirq.X(a)
    p2 = p.inplace_after(NoOp(a, b))
    assert p2 is p and p == cirq.X(a)

    # After H and back.
    p2 = p.inplace_after(cirq.H(a))
    assert p2 is p and p == cirq.Z(a)
    p2 = p.inplace_before(cirq.H(a))
    assert p2 is p and p == cirq.X(a)

    # After S and back.
    p2 = p.inplace_after(cirq.S(a))
    assert p2 is p and p == cirq.Y(a)
    p2 = p.inplace_before(cirq.S(a))
    assert p2 is p and p == cirq.X(a)

    # Before S and back.
    p2 = p.inplace_before(cirq.S(a))
    assert p2 is p and p == -cirq.Y(a)
    p2 = p.inplace_after(cirq.S(a))
    assert p2 is p and p == cirq.X(a)

    # After inverse S and back.
    p2 = p.inplace_after(cirq.S(a) ** -1)
    assert p2 is p and p == -cirq.Y(a)
    p2 = p.inplace_before(cirq.S(a) ** -1)
    assert p2 is p and p == cirq.X(a)

    # On other qubit.
    p2 = p.inplace_after(cirq.S(b))
    assert p2 is p and p == cirq.X(a)

    # Two qubit operation.
    p2 = p.inplace_after(cirq.CZ(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b)
    p2 = p.inplace_after(cirq.CZ(a, c))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b) * cirq.Z(c)
    p2 = p.inplace_after(cirq.H(b))
    assert p2 is p and p == cirq.X(a) * cirq.X(b) * cirq.Z(c)
    p2 = p.inplace_after(cirq.CNOT(b, c))
    assert p2 is p and p == -cirq.X(a) * cirq.Y(b) * cirq.Y(c)

    # Inverted interactions.
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Y, True, cirq.Z, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Z(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.X, False, cirq.Z, True).on(a, b))
    assert p2 is p and p == cirq.X(a)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Y, False, cirq.Z, True).on(a, b))
    assert p2 is p and p == -cirq.X(a) * cirq.Z(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, False, cirq.Y, True).on(a, b))
    assert p2 is p and p == -cirq.X(a) * cirq.Y(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, True, cirq.X, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.X(b)
    p = cirq.MutablePauliString(cirq.X(a))
    p2 = p.inplace_after(cirq.PauliInteractionGate(cirq.Z, True, cirq.Y, False).on(a, b))
    assert p2 is p and p == cirq.X(a) * cirq.Y(b)


def test_after_before_vs_conjugate_by():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.X(a) * cirq.Y(b) * cirq.Z(c)
    assert p.before(cirq.S(b)) == p.conjugated_by(cirq.S(b))
    assert p.after(cirq.S(b) ** -1) == p.conjugated_by(cirq.S(b))
    assert (
        p.before(cirq.CNOT(a, b)) == p.conjugated_by(cirq.CNOT(a, b)) == (p.after(cirq.CNOT(a, b)))
    )


def test_mutable_pauli_string_dict_functionality():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.MutablePauliString()
    with pytest.raises(KeyError):
        _ = p[a]
    assert p.get(a) is None
    assert a not in p
    assert not bool(p)
    p[a] = cirq.X
    assert bool(p)
    assert a in p
    assert p[a] == cirq.X

    p[a] = "Y"
    assert p[a] == cirq.Y
    p[a] = 3
    assert p[a] == cirq.Z
    p[a] = "I"
    assert a not in p
    p[a] = 0
    assert a not in p

    assert len(p) == 0
    p[b] = "Y"
    p[a] = "X"
    p[c] = "Z"
    assert len(p) == 3
    assert list(iter(p)) == [b, a, c]
    assert list(p.values()) == [cirq.Y, cirq.X, cirq.Z]
    assert list(p.keys()) == [b, a, c]
    assert p.keys() == {a, b, c}
    assert p.keys() ^ {c} == {a, b}

    del p[b]
    assert b not in p


def test_mutable_pauli_string_text():
    p = cirq.MutablePauliString(cirq.X(cirq.LineQubit(0)) * cirq.Y(cirq.LineQubit(1)))
    assert str(cirq.MutablePauliString()) == "mutable I"
    assert str(p) == "mutable X(0)*Y(1)"
    cirq.testing.assert_equivalent_repr(p)


def test_mutable_pauli_string_mul():
    a, b = cirq.LineQubit.range(2)
    p = cirq.X(a).mutable_copy()
    q = cirq.Y(b).mutable_copy()
    pq = cirq.X(a) * cirq.Y(b)
    assert p * q == pq
    assert isinstance(p * q, cirq.PauliString)
    assert 2 * p == cirq.X(a) * 2 == p * 2
    assert isinstance(p * 2, cirq.PauliString)
    assert isinstance(2 * p, cirq.PauliString)


def test_mutable_can_override_mul():
    class LMul:
        def __mul__(self, other):
            return "Yay!"

    class RMul:
        def __rmul__(self, other):
            return "Yay!"

    assert cirq.MutablePauliString() * RMul() == "Yay!"
    assert LMul() * cirq.MutablePauliString() == "Yay!"


def test_coefficient_precision():
    qs = cirq.LineQubit.range(4 * 10 ** 3)
    r = cirq.MutablePauliString({q: cirq.X for q in qs})
    r2 = cirq.MutablePauliString({q: cirq.Y for q in qs})
    r2 *= r
    assert r2.coefficient == 1


def test_transform_qubits():
    a, b, c = cirq.LineQubit.range(3)
    p = cirq.X(a) * cirq.Z(b)
    p2 = cirq.X(b) * cirq.Z(c)
    m = p.mutable_copy()
    m2 = m.transform_qubits(lambda q: q + 1)
    assert m is not m2
    assert m == p
    assert m2 == p2

    m2 = m.transform_qubits(lambda q: q + 1, inplace=False)
    assert m is not m2
    assert m == p
    assert m2 == p2

    m2 = m.transform_qubits(lambda q: q + 1, inplace=True)
    assert m is m2
    assert m == p2
    assert m2 == p2
