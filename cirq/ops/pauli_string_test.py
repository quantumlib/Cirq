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
        lambda: cirq.PauliString({}),
        lambda: cirq.PauliString({}, +1))
    eq.add_equality_group(cirq.PauliString({}, -1))
    for q, pauli in itertools.product((q0, q1), (cirq.X, cirq.Y, cirq.Z)):
        eq.add_equality_group(cirq.PauliString({q: pauli}, +1))
        eq.add_equality_group(cirq.PauliString({q: pauli}, -1))
    for q, p0, p1 in itertools.product((q0, q1), (cirq.X, cirq.Y, cirq.Z),
                                       (cirq.X, cirq.Y, cirq.Z)):
        eq.add_equality_group(cirq.PauliString({q: p0, q2: p1}, +1))


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

    with pytest.raises(NotImplementedError, match='non-hermitian'):
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

    with pytest.raises(NotImplementedError, match='non-unitary'):
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
def test_from_single(pauli):
    q0, = _make_qubits(1)
    assert (cirq.PauliString.from_single(q0, pauli)
            == cirq.PauliString({q0: pauli}))


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_getitem(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = cirq.PauliString(qubit_pauli_map)
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

    assert cirq.PauliString.from_single(q0, cirq.X).commutes_with(
           cirq.PauliString.from_single(q0, cirq.X))
    assert not cirq.PauliString.from_single(q0, cirq.X).commutes_with(
               cirq.PauliString.from_single(q0, cirq.Y))
    assert cirq.PauliString.from_single(q0, cirq.X).commutes_with(
           cirq.PauliString.from_single(q1, cirq.X))
    assert cirq.PauliString.from_single(q0, cirq.X).commutes_with(
           cirq.PauliString.from_single(q1, cirq.Y))

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
    with pytest.raises(TypeError):
        _ = p * 'test'
    with pytest.raises(TypeError):
        _ = 'test' * p


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
        cirq.PauliString.from_single(a, cirq.X),
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
    with pytest.raises(NotImplementedError, match="non-hermitian"):
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
    circuit = cirq.Circuit.from_ops(
                    pauli_string.to_z_basis_ops())

    initial_state = cirq.kron(x0, x1, y0, y1, z0, z1)
    z_basis_state = circuit.apply_unitary_effect_to_state(initial_state)

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
