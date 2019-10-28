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

import cirq
from cirq._compat_test import capture_logging


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]


def _sample_qubit_pauli_maps():
    qubits = _make_qubits(3)
    paulis_or_none = (None, cirq.X, cirq.Y, cirq.Z)
    for paulis in itertools.product(paulis_or_none, repeat=len(qubits)):
        yield {qubit: pauli for qubit, pauli in zip(qubits, paulis)
                            if pauli is not None}


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(
        lambda: cirq.PauliString(), lambda: cirq.PauliString(qubit_pauli_map={
        }), lambda: cirq.PauliString(qubit_pauli_map={}, coefficient=+1))
    eq.add_equality_group(cirq.PauliString(qubit_pauli_map={}, coefficient=-1))
    for q, pauli in itertools.product((q0, q1), (cirq.X, cirq.Y, cirq.Z)):
        eq.add_equality_group(
            cirq.PauliString(qubit_pauli_map={q: pauli}, coefficient=+1))
        eq.add_equality_group(
            cirq.PauliString(qubit_pauli_map={q: pauli}, coefficient=-1))
    for q, p0, p1 in itertools.product((q0, q1), (cirq.X, cirq.Y, cirq.Z),
                                       (cirq.X, cirq.Y, cirq.Z)):
        eq.add_equality_group(
            cirq.PauliString(qubit_pauli_map={
                q: p0,
                q2: p1
            }, coefficient=+1))


def test_equal_up_to_coefficient():
    q0, = _make_qubits(1)
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(
           cirq.PauliString({}, +1))
    assert cirq.PauliString({}, -1).equal_up_to_coefficient(
           cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(
           cirq.PauliString({}, -1))
    assert cirq.PauliString({}, +1).equal_up_to_coefficient(
           cirq.PauliString({}, 2j))

    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
           cirq.PauliString({q0: cirq.X}, +1))
    assert cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(
           cirq.PauliString({q0: cirq.X}, -1))
    assert cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
           cirq.PauliString({q0: cirq.X}, -1))

    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
               cirq.PauliString({q0: cirq.Y}, +1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
               cirq.PauliString({q0: cirq.Y}, 1j))
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(
               cirq.PauliString({q0: cirq.Y}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
               cirq.PauliString({q0: cirq.Y}, -1))

    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
               cirq.PauliString({}, +1))
    assert not cirq.PauliString({q0: cirq.X}, -1).equal_up_to_coefficient(
               cirq.PauliString({}, -1))
    assert not cirq.PauliString({q0: cirq.X}, +1).equal_up_to_coefficient(
               cirq.PauliString({}, -1))


def test_exponentiation_as_exponent():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})

    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        _ = math.e**(math.pi * p)

    with pytest.raises(TypeError, match='unsupported'):
        _ = 'test'**p

    assert cirq.approx_eq(
        math.e**(-1j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=-0.5))

    assert cirq.approx_eq(
        math.e**(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))

    assert cirq.approx_eq(
        2**(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p,
                               exponent_neg=-0.25 * math.log(2),
                               exponent_pos=0.25 * math.log(2)))

    assert cirq.approx_eq(
        np.exp(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))


def test_exponentiate_single_value_as_exponent():
    q = cirq.LineQubit(0)

    assert cirq.approx_eq(math.e**(-0.25j * math.pi * cirq.X(q)),
                          cirq.Rx(0.25 * math.pi).on(q))

    assert cirq.approx_eq(math.e**(-0.25j * math.pi * cirq.Y(q)),
                          cirq.Ry(0.25 * math.pi).on(q))

    assert cirq.approx_eq(math.e**(-0.25j * math.pi * cirq.Z(q)),
                          cirq.Rz(0.25 * math.pi).on(q))

    assert cirq.approx_eq(np.exp(-0.3j * math.pi * cirq.X(q)),
                          cirq.Rx(0.3 * math.pi).on(q))

    assert cirq.approx_eq(cirq.X(q)**0.5, cirq.XPowGate(exponent=0.5).on(q))

    assert cirq.approx_eq(cirq.Y(q)**0.5, cirq.YPowGate(exponent=0.5).on(q))

    assert cirq.approx_eq(cirq.Z(q)**0.5, cirq.ZPowGate(exponent=0.5).on(q))


def test_exponentiation_as_base():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})

    with pytest.raises(TypeError, match='unsupported'):
        _ = (2 * p)**5

    with pytest.raises(TypeError, match='unsupported'):
        _ = p**'test'

    with pytest.raises(TypeError, match='unsupported'):
        _ = p**1j

    assert p**-1 == p

    assert cirq.approx_eq(
        p**0.5, cirq.PauliStringPhasor(p, exponent_neg=0.5, exponent_pos=0))

    assert cirq.approx_eq(
        p**-0.5, cirq.PauliStringPhasor(p, exponent_neg=-0.5, exponent_pos=0))

    assert cirq.approx_eq(
        math.e**(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))

    assert cirq.approx_eq(
        2**(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p,
                               exponent_neg=-0.25 * math.log(2),
                               exponent_pos=0.25 * math.log(2)))

    assert cirq.approx_eq(
        np.exp(0.5j * math.pi * p),
        cirq.PauliStringPhasor(p, exponent_neg=-0.25, exponent_pos=0.25))


@pytest.mark.parametrize('pauli', (cirq.X, cirq.Y, cirq.Z))
def test_list_op_constructor_matches_mapping(pauli):
    q0, = _make_qubits(1)
    op = pauli.on(q0)
    assert cirq.PauliString([op]) == cirq.PauliString({q0: pauli})


def test_constructor_flexibility():
    a, b = cirq.LineQubit.range(2)
    with pytest.raises(TypeError, match='Not a `cirq.PAULI_STRING_LIKE`'):
        _ = cirq.PauliString(cirq.CZ(a, b))
    with pytest.raises(TypeError, match='Not a `cirq.PAULI_STRING_LIKE`'):
        _ = cirq.PauliString('test')
    with pytest.raises(TypeError, match='S is not a Pauli'):
        _ = cirq.PauliString(qubit_pauli_map={a: cirq.S})

    assert cirq.PauliString(
        cirq.X(a)) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([cirq.X(a)
                            ]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.X(a)]]
                            ]) == cirq.PauliString(qubit_pauli_map={a: cirq.X})
    assert cirq.PauliString([[[cirq.I(a)]]]) == cirq.PauliString()

    assert cirq.PauliString(1, 2, 3, cirq.X(a), cirq.Y(a)) == cirq.PauliString(
        qubit_pauli_map={a: cirq.Z}, coefficient=6j)

    assert cirq.PauliString(cirq.X(a), cirq.X(a)) == cirq.PauliString()
    assert cirq.PauliString(cirq.X(a),
                            cirq.X(b)) == cirq.PauliString(qubit_pauli_map={
                                a: cirq.X,
                                b: cirq.X
                            })

    assert cirq.PauliString(0) == cirq.PauliString(coefficient=0)

    assert cirq.PauliString(1, 2, 3, {a: cirq.X},
                            cirq.Y(a)) == cirq.PauliString(
                                qubit_pauli_map={a: cirq.Z}, coefficient=6j)


def test_deprecated_from_single():
    q0 = cirq.LineQubit(0)
    with capture_logging() as log:
        actual = cirq.PauliString.from_single(q0, cirq.X)
    assert len(log) == 1  # May fail if deprecated thing is used elsewhere.
    assert 'PauliString.from_single' in log[0].getMessage()
    assert 'deprecated' in log[0].getMessage()

    assert actual == cirq.PauliString([cirq.X(q0)])


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
    assert qubit_pauli_map.get(other) == pauli_string.get(other) == None
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
def test_keys(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    assert (len(qubit_pauli_map.keys()) == len(pauli_string.keys())
            == len(pauli_string.qubits))
    assert (set(qubit_pauli_map.keys()) == set(pauli_string.keys())
            == set(pauli_string.qubits))


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_items(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map.items()) == len(pauli_string.items())
    assert set(qubit_pauli_map.items()) == set(pauli_string.items())


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_values(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map.values()) == len(pauli_string.values())
    assert set(qubit_pauli_map.values()) == set(pauli_string.values())


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_len(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map) == len(pauli_string)


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_iter(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
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
    assert str(1j*pauli_string) == '1j*Z(q0)*Y(q1)*X(q2)'
    assert str(pauli_string*-1j) == '-1j*Z(q0)*Y(q1)*X(q2)'


@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, {}),
        ({q0: cirq.X}, {q0: cirq.Y}, {q0: (cirq.X, cirq.Y)}),
        ({q0: cirq.X}, {q1: cirq.X}, {}),
        ({q0: cirq.Y, q1: cirq.Z}, {q1: cirq.Y, q2: cirq.X},
            {q1: (cirq.Z, cirq.Y)}),
        ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {}, {}),
        ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {q0: cirq.Y, q1: cirq.Z},
            {q0: (cirq.X, cirq.Y), q1: (cirq.Y, cirq.Z)}),
    ))(*_make_qubits(3)))
def test_zip_items(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_items(ps2))
    assert len(out_actual) == len(out)
    assert dict(out_actual) == out


@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, ()),
        ({q0: cirq.X}, {q0: cirq.Y}, ((cirq.X, cirq.Y),)),
        ({q0: cirq.X}, {q1: cirq.X}, ()),
        ({q0: cirq.Y, q1: cirq.Z}, {q1: cirq.Y, q2: cirq.X},
            ((cirq.Z, cirq.Y),)),
        ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {}, ()),
        ({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}, {q0: cirq.Y, q1: cirq.Z},
            # Order not necessary
            ((cirq.X, cirq.Y), (cirq.Y, cirq.Z)))
    ))(*_make_qubits(3)))
def test_zip_paulis(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_paulis(ps2))
    assert len(out_actual) == len(out)
    if len(out) <= 1:
        assert out_actual == out
    assert set(out_actual) == set(out)  # Ignore output order


def test_commutes_with():
    q0, q1, q2 = _make_qubits(3)

    assert cirq.PauliString([cirq.X.on(q0)
                            ]).commutes_with(cirq.PauliString([cirq.X.on(q0)]))
    assert not cirq.PauliString([cirq.X.on(q0)]).commutes_with(
        cirq.PauliString([cirq.Y.on(q0)]))
    assert cirq.PauliString([cirq.X.on(q0)
                            ]).commutes_with(cirq.PauliString([cirq.X.on(q1)]))
    assert cirq.PauliString([cirq.X.on(q0)
                            ]).commutes_with(cirq.PauliString([cirq.Y.on(q1)]))

    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.X, q1: cirq.Y}))
    assert not cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
               cirq.PauliString({q0: cirq.X, q1: cirq.Z}))
    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Y, q1: cirq.X}))
    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Y, q1: cirq.Z}))

    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.X, q1: cirq.Y, q2: cirq.Z}))
    assert not cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
               cirq.PauliString({q0: cirq.X, q1: cirq.Z, q2: cirq.Z}))
    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Y, q1: cirq.X, q2: cirq.Z}))
    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Y, q1: cirq.Z, q2: cirq.X}))

    assert cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
           cirq.PauliString({q2: cirq.X, q1: cirq.Y}))
    assert not cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
               cirq.PauliString({q2: cirq.X, q1: cirq.Z}))
    assert not cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
               cirq.PauliString({q2: cirq.Y, q1: cirq.X}))
    assert not cirq.PauliString({q0: cirq.X, q1: cirq.Y}).commutes_with(
               cirq.PauliString({q2: cirq.Y, q1: cirq.Z}))


def test_negate():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.X, q1: cirq.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    ps2 = cirq.PauliString(qubit_pauli_map, -1)
    assert -ps1 == ps2
    assert ps1 == -ps2
    neg_ps1 = -ps1
    assert -neg_ps1 == ps1


def test_mul_scalar():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert -p == -1 * p == -1.0 * p == p * -1 == p * complex(-1)
    assert -p != 1j * p
    assert +p == 1 * p

    assert p * cirq.I(a) == p
    assert cirq.I(a) * p == p

    with pytest.raises(TypeError,
                       match="sequence by non-int of type 'PauliString'"):
        _ = p * 'test'
    with pytest.raises(TypeError,
                       match="sequence by non-int of type 'PauliString'"):
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
        _ = 'test' / p


def test_mul_strings():
    a, b, c, d = cirq.LineQubit.range(4)
    p1 = cirq.PauliString({a: cirq.X, b: cirq.Y, c: cirq.Z})
    p2 = cirq.PauliString({b: cirq.X, c: cirq.Y, d: cirq.Z})
    assert p1 * p2 == -cirq.PauliString({
        a: cirq.X,
        b: cirq.Z,
        c: cirq.X,
        d: cirq.Z,
    })

    assert cirq.X(a) * cirq.PauliString({a: cirq.X}) == cirq.PauliString()
    assert cirq.PauliString({a: cirq.X}) * cirq.X(a) == cirq.PauliString()
    assert cirq.X(a) * cirq.X(a) == cirq.PauliString()
    assert -cirq.X(a) * -cirq.X(a) == cirq.PauliString()

    with pytest.raises(TypeError, match='unsupported'):
        _ = cirq.X(a) * object()
    with pytest.raises(TypeError, match='unsupported'):
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
    assert cirq.Y(a) * cirq.Z(b) * cirq.X(a) == -1j * cirq.PauliString({
        a: cirq.Z,
        b: cirq.Z
    })


def test_pos():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.X, q1: cirq.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    assert ps1 == +ps1


def test_pow():
    a, b = cirq.LineQubit.range(2)

    assert cirq.PauliString({a: cirq.X})**0.25 == cirq.X(a)**0.25
    assert cirq.PauliString({a: cirq.Y})**0.25 == cirq.Y(a)**0.25
    assert cirq.PauliString({a: cirq.Z})**0.25 == cirq.Z(a)**0.25

    p = cirq.PauliString({a: cirq.X, b: cirq.Y})
    assert p**1 == p
    assert p**-1 == p
    assert (-p)**1 == -p
    assert (-p)**-1 == -p
    assert (1j * p)**1 == 1j * p
    assert (1j * p)**-1 == -1j * p


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
    x0 = np.array([1,1]) / np.sqrt(2)
    x1 = np.array([1,-1]) / np.sqrt(2)
    y0 = np.array([1,1j]) / np.sqrt(2)
    y1 = np.array([1,-1j]) / np.sqrt(2)
    z0 = np.array([1,0])
    z1 = np.array([0,1])

    q0, q1, q2, q3, q4, q5 = _make_qubits(6)
    pauli_string = cirq.PauliString({q0: cirq.X, q1: cirq.X,
                                     q2: cirq.Y, q3: cirq.Y,
                                     q4: cirq.Z, q5: cirq.Z})
    circuit = cirq.Circuit(pauli_string.to_z_basis_ops())

    initial_state = cirq.kron(x0, x1, y0, y1, z0, z1)
    z_basis_state = circuit.final_wavefunction(initial_state)

    expected_state = np.zeros(2 ** 6)
    expected_state[0b010101] = 1

    cirq.testing.assert_allclose_up_to_global_phase(
                    z_basis_state, expected_state, rtol=1e-7, atol=1e-7)


def _assert_pass_over(ops: List[cirq.Operation],
                      before: cirq.PauliString,
                      after: cirq.PauliString):
    assert before.pass_operations_over(ops[::-1]) == after
    assert after.pass_operations_over(ops, after_to_before=True) == before


@pytest.mark.parametrize('shift,sign',
                         itertools.product(range(3), (-1, +1)))
def test_pass_operations_over_single(shift: int, sign: int):
    q0, q1 = _make_qubits(2)
    X, Y, Z = (cirq.Pauli.by_relative_index(cast(cirq.Pauli, pauli), shift)
               for pauli in (cirq.X, cirq.Y, cirq.Z))

    op0 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X}, sign)
    ps_after = ps_before
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(X)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, sign)
    ps_after = ps_before
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_double_map({Z: (X, False),
                                                        X: (Z, False)})(q0)
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


@pytest.mark.parametrize('shift,t_or_f1, t_or_f2,neg',
        itertools.product(range(3), *((True, False),)*3))
def test_pass_operations_over_double(shift, t_or_f1, t_or_f2, neg):
    sign = -1 if neg else +1
    q0, q1, q2 = _make_qubits(3)
    X, Y, Z = (cirq.Pauli.by_relative_index(pauli, shift)
               for pauli in (cirq.X, cirq.Y, cirq.Z))

    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q2: Y}, sign)
    ps_after = cirq.PauliString({q0: Z, q2: Y}, sign)
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
    ps_after = cirq.PauliString({q0: X, q1: Z},
                                -1 if neg ^ t_or_f1 ^ t_or_f2 else +1)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(X, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, sign)
    ps_after = cirq.PauliString({q0: Y, q1: Z},
                                +1 if neg ^ t_or_f1 ^ t_or_f2 else -1)
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
    q0, = _make_qubits(1)
    pauli_string = cirq.PauliString({q0: cirq.X})
    with pytest.raises(TypeError):
        pauli_string.pass_operations_over([cirq.X(q0)])


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


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_consistency(qubit_pauli_map):
    pauli_string = cirq.PauliString(qubit_pauli_map)
    cirq.testing.assert_implements_consistent_protocols(pauli_string)


def test_scaled_unitary_consistency():
    a, b = cirq.LineQubit.range(2)
    cirq.testing.assert_implements_consistent_protocols(2 * cirq.X(a) *
                                                        cirq.Y(b))
    cirq.testing.assert_implements_consistent_protocols(1j * cirq.X(a) *
                                                        cirq.Y(b))


def test_bool():
    a = cirq.LineQubit(0)
    assert not bool(cirq.PauliString({}))
    assert bool(cirq.PauliString({a: cirq.X}))


def test_unitary_matrix():
    a, b = cirq.LineQubit.range(2)
    assert not cirq.has_unitary(2 * cirq.X(a) * cirq.Z(b))
    assert cirq.unitary(2 * cirq.X(a) * cirq.Z(b), default=None) is None
    np.testing.assert_allclose(
        cirq.unitary(cirq.X(a) * cirq.Z(b)),
        np.array([
            [0, 0, 1, 0],
            [0, 0, 0, -1],
            [1, 0, 0, 0],
            [0, -1, 0, 0],
        ]))
    np.testing.assert_allclose(
        cirq.unitary(1j * cirq.X(a) * cirq.Z(b)),
        np.array([
            [0, 0, 1j, 0],
            [0, 0, 0, -1j],
            [1j, 0, 0, 0],
            [0, -1j, 0, 0],
        ]))


def test_decompose():
    a, b = cirq.LineQubit.range(2)
    assert cirq.decompose_once(2 * cirq.X(a) * cirq.Z(b), default=None) is None
    assert cirq.decompose_once(1j * cirq.X(a) * cirq.Z(b)) == [
        cirq.GlobalPhaseOperation(1j),
        cirq.X(a), cirq.Z(b)
    ]
    assert cirq.decompose_once(cirq.Y(b) * cirq.Z(a)) == [cirq.Z(a), cirq.Y(b)]


def test_rejects_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.PauliString({q: cirq.S})


def test_cannot_multiply_by_non_paulis():
    q = cirq.NamedQubit('q')
    with pytest.raises(TypeError):
        _ = cirq.X(q) * cirq.Z(q)**0.5
    with pytest.raises(TypeError):
        _ = cirq.Z(q)**0.5 * cirq.X(q)
    with pytest.raises(TypeError):
        _ = cirq.Y(q) * cirq.S(q)


def test_filters_identities():
    q1, q2 = cirq.LineQubit.range(2)
    assert cirq.PauliString({q1: cirq.I, q2: cirq.X}) == \
           cirq.PauliString({q2: cirq.X})


def test_expectation_from_wavefunction_invalid_input():
    q0, q1, q2, q3 = _make_qubits(4)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    wf = np.array([1, 0, 0, 0], dtype=np.complex64)
    q_map = {q0: 0, q1: 1}

    im_ps = (1j + 1) * ps
    with pytest.raises(NotImplementedError, match='non-Hermitian'):
        im_ps.expectation_from_wavefunction(wf, q_map)

    with pytest.raises(TypeError, match='dtype'):
        ps.expectation_from_wavefunction(np.array([1, 0], dtype=np.int), q_map)

    with pytest.raises(TypeError, match='mapping'):
        ps.expectation_from_wavefunction(wf, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        ps.expectation_from_wavefunction(wf, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
        ps.expectation_from_wavefunction(wf, {q0: "bad value"})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_wavefunction(wf, {q0: 0})
    with pytest.raises(ValueError, match='complete'):
        ps.expectation_from_wavefunction(wf, {q0: 0, q2: 2})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_wavefunction(wf, {q0: -1, q1: 1})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_wavefunction(wf, {q0: 0, q1: 3})
    with pytest.raises(ValueError, match='indices'):
        ps.expectation_from_wavefunction(wf, {q0: 0, q1: 0})
    # Excess keys are ignored.
    _ = ps.expectation_from_wavefunction(wf, {q0: 0, q1: 1, q2: 0})

    # Incorrectly shaped wavefunction input.
    with pytest.raises(ValueError, match='7'):
        ps.expectation_from_wavefunction(np.arange(7, dtype=np.complex64),
                                         q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='normalized'):
        ps.expectation_from_wavefunction(np.arange(16, dtype=np.complex64),
                                         q_map_2)

    # The ambiguous case: Density matrices satisfying L2 normalization.
    rho_or_wf = 0.5 * np.ones((2, 2), dtype=np.complex64)
    _ = ps.expectation_from_wavefunction(rho_or_wf, q_map)

    wf = np.arange(16, dtype=np.complex64) / np.linalg.norm(np.arange(16))
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_wavefunction(wf.reshape((16, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_wavefunction(wf.reshape((4, 4, 1)), q_map_2)


def test_expectation_from_wavefunction_check_preconditions():
    q0, q1, q2, q3 = _make_qubits(4)
    ps = cirq.PauliString({q0: cirq.X, q1: cirq.Y})
    q_map = {q0: 0, q1: 1, q2: 2, q3: 3}

    with pytest.raises(ValueError, match='normalized'):
        ps.expectation_from_wavefunction(np.arange(16, dtype=np.complex64),
                                         q_map)

    _ = ps.expectation_from_wavefunction(np.arange(16, dtype=np.complex64),
                                         q_map,
                                         check_preconditions=False)


def test_expectation_from_wavefunction_basis_states():
    q0 = cirq.LineQubit(0)
    x0 = cirq.PauliString({q0: cirq.X})
    q_map = {q0: 0}

    np.testing.assert_allclose(
        x0.expectation_from_wavefunction(np.array([1, 0], dtype=np.complex),
                                         q_map), 0)
    np.testing.assert_allclose(
        x0.expectation_from_wavefunction(np.array([0, 1], dtype=np.complex),
                                         q_map), 0)
    np.testing.assert_allclose(
        x0.expectation_from_wavefunction(
            np.array([1, 1], dtype=np.complex) / np.sqrt(2), q_map), 1)
    np.testing.assert_allclose(
        x0.expectation_from_wavefunction(
            np.array([1, -1], dtype=np.complex) / np.sqrt(2), q_map), -1)

    y0 = cirq.PauliString({q0: cirq.Y})
    np.testing.assert_allclose(
        y0.expectation_from_wavefunction(
            np.array([1, 1j], dtype=np.complex) / np.sqrt(2), q_map), 1)
    np.testing.assert_allclose(
        y0.expectation_from_wavefunction(
            np.array([1, -1j], dtype=np.complex) / np.sqrt(2), q_map), -1)
    np.testing.assert_allclose(
        y0.expectation_from_wavefunction(
            np.array([1, 1], dtype=np.complex) / np.sqrt(2), q_map), 0)
    np.testing.assert_allclose(
        y0.expectation_from_wavefunction(
            np.array([1, -1], dtype=np.complex) / np.sqrt(2), q_map), 0)


def test_expectation_from_wavefunction_entangled_states():
    q0, q1 = _make_qubits(2)
    z0z1_pauli_map = {q0: cirq.Z, q1: cirq.Z}
    z0z1 = cirq.PauliString(z0z1_pauli_map)
    x0x1_pauli_map = {q0: cirq.X, q1: cirq.X}
    x0x1 = cirq.PauliString(x0x1_pauli_map)
    q_map = {q0: 0, q1: 1}
    wf1 = np.array([0, 1, 1, 0], dtype=np.complex) / np.sqrt(2)
    for state in [wf1, wf1.reshape(2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_wavefunction(state, q_map), -1)
        np.testing.assert_allclose(
            x0x1.expectation_from_wavefunction(state, q_map), 1)

    wf2 = np.array([1, 0, 0, 1], dtype=np.complex) / np.sqrt(2)
    for state in [wf2, wf2.reshape(2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_wavefunction(state, q_map), 1)
        np.testing.assert_allclose(
            x0x1.expectation_from_wavefunction(state, q_map), 1)

    wf3 = np.array([1, 1, 1, 1], dtype=np.complex) / 2
    for state in [wf3, wf3.reshape(2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            x0x1.expectation_from_wavefunction(state, q_map), 1)


def test_expectation_from_wavefunction_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.complex) / np.sqrt(2)
    for state in [wf, wf.reshape(2, 2, 2)]:
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 0,
                q1: 1,
                q2: 2
            }), 1)
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 0,
                q1: 2,
                q2: 1
            }), 1)
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 1,
                q1: 0,
                q2: 2
            }), 0)
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 1,
                q1: 2,
                q2: 0
            }), 0)
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 2,
                q1: 0,
                q2: 1
            }), -1)
        np.testing.assert_allclose(
            z.expectation_from_wavefunction(state, {
                q0: 2,
                q1: 1,
                q2: 0
            }), -1)


def test_pauli_string_expectation_from_wavefunction_pure_state():
    qubits = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.H(qubits[2]),
        cirq.X(qubits[3]),
        cirq.H(qubits[3]),
    )
    wf = circuit.final_wavefunction(qubit_order=qubits)

    z0z1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
    z0z2 = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})
    z0z3 = cirq.PauliString({qubits[0]: cirq.Z, qubits[3]: cirq.Z})
    z0x1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.X})
    z1x2 = cirq.PauliString({qubits[1]: cirq.Z, qubits[2]: cirq.X})
    x0z1 = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    x3 = cirq.PauliString({qubits[3]: cirq.X})

    for state in [wf, wf.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_wavefunction(state, q_map), -1)
        np.testing.assert_allclose(
            z0z2.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            z0z3.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            z0x1.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            z1x2.expectation_from_wavefunction(state, q_map), -1)
        np.testing.assert_allclose(
            x0z1.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            x3.expectation_from_wavefunction(state, q_map), -1)


def test_pauli_string_expectation_from_wavefunction_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qs)}

    circuit = cirq.Circuit(
        cirq.X(qs[1]),
        cirq.H(qs[2]),
        cirq.X(qs[3]),
        cirq.H(qs[3]),
    )
    wf = circuit.final_wavefunction(qubit_order=qs)

    z0z1 = cirq.Z(qs[0]) * cirq.Z(qs[1]) * .123
    z0z2 = cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1
    z1x2 = -cirq.Z(qs[1]) * cirq.X(qs[2])

    for state in [wf, wf.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_wavefunction(state, q_map), -0.123)
        np.testing.assert_allclose(
            z0z2.expectation_from_wavefunction(state, q_map), 0)
        np.testing.assert_allclose(
            z1x2.expectation_from_wavefunction(state, q_map), 1)


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
        ps.expectation_from_density_matrix(rho, "bad type")
    with pytest.raises(TypeError, match='mapping'):
        ps.expectation_from_density_matrix(rho, {"bad key": 1})
    with pytest.raises(TypeError, match='mapping'):
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
                [[1.1, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.complex64), q_map)

    # Incorrectly shaped density matrix input.
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(np.ones((4, 5), dtype=np.complex64),
                                           q_map)
    q_map_2 = {q0: 0, q1: 1, q2: 2, q3: 3}
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(rho.reshape((4, 4, 1)), q_map_2)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(rho.reshape((-1)), q_map_2)

    # Correctly shaped wavefunctions.
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(np.array([1, 0], dtype=np.complex64),
                                           q_map)
    with pytest.raises(ValueError, match='shape'):
        ps.expectation_from_density_matrix(wf, q_map)

    # The ambiguous cases: Wavefunctions satisfying trace normalization.
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
                [[1.1, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                dtype=np.complex64), q_map)

    _ = ps.expectation_from_density_matrix(np.array(
        [[1.1, 0, 0, 0], [0, -.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=np.complex64),
                                           q_map,
                                           check_preconditions=False)


def test_expectation_from_density_matrix_basis_states():
    q0 = cirq.LineQubit(0)
    x0_pauli_map = {q0: cirq.X}
    x0 = cirq.PauliString(x0_pauli_map)
    q_map = {q0: 0}
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(
            np.array([[1, 0], [0, 0]], dtype=np.complex), q_map), 0)
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(
            np.array([[0, 0], [0, 1]], dtype=np.complex), q_map), 0)
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(
            np.array([[1, 1], [1, 1]], dtype=np.complex) / 2, q_map), 1)
    np.testing.assert_allclose(
        x0.expectation_from_density_matrix(
            np.array([[1, -1], [-1, 1]], dtype=np.complex) / 2, q_map), -1)


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
        np.testing.assert_allclose(
            z0z1.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(
            x0x1.expectation_from_density_matrix(state, q_map), 1)

    wf2 = np.array([1, 0, 0, 1], dtype=np.complex) / np.sqrt(2)
    rho2 = np.kron(wf2, wf2).reshape(4, 4)
    for state in [rho2, rho2.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_density_matrix(state, q_map), 1)
        np.testing.assert_allclose(
            x0x1.expectation_from_density_matrix(state, q_map), 1)

    wf3 = np.array([1, 1, 1, 1], dtype=np.complex) / 2
    rho3 = np.kron(wf3, wf3).reshape(4, 4)
    for state in [rho3, rho3.reshape(2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            x0x1.expectation_from_density_matrix(state, q_map), 1)


def test_expectation_from_density_matrix_qubit_map():
    q0, q1, q2 = _make_qubits(3)
    z = cirq.PauliString({q0: cirq.Z})
    wf = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.complex) / np.sqrt(2)
    rho = np.kron(wf, wf).reshape(8, 8)

    for state in [rho, rho.reshape(2, 2, 2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 0,
                q1: 1,
                q2: 2
            }), 1)
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 0,
                q1: 2,
                q2: 1
            }), 1)
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 1,
                q1: 0,
                q2: 2
            }), 0)
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 1,
                q1: 2,
                q2: 0
            }), 0)
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 2,
                q1: 0,
                q2: 1
            }), -1)
        np.testing.assert_allclose(
            z.expectation_from_density_matrix(state, {
                q0: 2,
                q1: 1,
                q2: 0
            }), -1)


def test_pauli_string_expectation_from_density_matrix_pure_state():
    qubits = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qubits)}

    circuit = cirq.Circuit(
        cirq.X(qubits[1]),
        cirq.H(qubits[2]),
        cirq.X(qubits[3]),
        cirq.H(qubits[3]),
    )
    wavefunction = circuit.final_wavefunction(qubit_order=qubits)
    rho = np.outer(wavefunction, np.conj(wavefunction))

    z0z1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.Z})
    z0z2 = cirq.PauliString({qubits[0]: cirq.Z, qubits[2]: cirq.Z})
    z0z3 = cirq.PauliString({qubits[0]: cirq.Z, qubits[3]: cirq.Z})
    z0x1 = cirq.PauliString({qubits[0]: cirq.Z, qubits[1]: cirq.X})
    z1x2 = cirq.PauliString({qubits[1]: cirq.Z, qubits[2]: cirq.X})
    x0z1 = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    x3 = cirq.PauliString({qubits[3]: cirq.X})

    for state in [rho, rho.reshape(2, 2, 2, 2, 2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(
            z0z2.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            z0z3.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            z0x1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            z1x2.expectation_from_density_matrix(state, q_map), -1)
        np.testing.assert_allclose(
            x0z1.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            x3.expectation_from_density_matrix(state, q_map), -1)


def test_pauli_string_expectation_from_density_matrix_pure_state_with_coef():
    qs = cirq.LineQubit.range(4)
    q_map = {q: i for i, q in enumerate(qs)}

    circuit = cirq.Circuit(
        cirq.X(qs[1]),
        cirq.H(qs[2]),
        cirq.X(qs[3]),
        cirq.H(qs[3]),
    )
    wavefunction = circuit.final_wavefunction(qubit_order=qs)
    rho = np.outer(wavefunction, np.conj(wavefunction))

    z0z1 = cirq.Z(qs[0]) * cirq.Z(qs[1]) * .123
    z0z2 = cirq.Z(qs[0]) * cirq.Z(qs[2]) * -1
    z1x2 = -cirq.Z(qs[1]) * cirq.X(qs[2])

    for state in [rho, rho.reshape(2, 2, 2, 2, 2, 2, 2, 2)]:
        np.testing.assert_allclose(
            z0z1.expectation_from_density_matrix(state, q_map), -0.123)
        np.testing.assert_allclose(
            z0z2.expectation_from_density_matrix(state, q_map), 0)
        np.testing.assert_allclose(
            z1x2.expectation_from_density_matrix(state, q_map), 1)


def test_pauli_string_expectation_from_wavefunction_mixed_state_linearity():
    n_qubits = 10

    wavefunction1 = cirq.testing.random_superposition(2**n_qubits)
    wavefunction2 = cirq.testing.random_superposition(2**n_qubits)
    rho1 = np.outer(wavefunction1, np.conj(wavefunction1))
    rho2 = np.outer(wavefunction2, np.conj(wavefunction2))
    density_matrix = rho1 / 2 + rho2 / 2

    qubits = cirq.LineQubit.range(n_qubits)
    q_map = {q: i for i, q in enumerate(qubits)}
    paulis = [cirq.X, cirq.Y, cirq.Z]
    pauli_string = cirq.PauliString(
        {q: np.random.choice(paulis) for q in qubits})

    a = pauli_string.expectation_from_wavefunction(wavefunction1, q_map)
    b = pauli_string.expectation_from_wavefunction(wavefunction2, q_map)
    c = pauli_string.expectation_from_density_matrix(density_matrix, q_map)
    np.testing.assert_allclose(0.5 * (a + b), c)
