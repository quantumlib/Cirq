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
import pytest
from cirq.testing import (
    EqualsTester,
)

import cirq

from cirq.contrib.paulistring import (
    Pauli,
    CliffordGate,
    PauliInteractionGate,
    PauliString,
)


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]

def _sample_qubit_pauli_maps():
    qubits = _make_qubits(3)
    yield {}
    paulis_or_none = (None,) + Pauli.XYZ
    for paulis in itertools.product(*(paulis_or_none,)*len(qubits)):
        yield {qubit: pauli for qubit, pauli in zip(qubits, paulis)
                            if pauli is not None}


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = EqualsTester()
    eq.make_equality_group(
        lambda: PauliString({}),
        lambda: PauliString({}, False))
    eq.make_equality_group(lambda: PauliString({}, True))
    for q, pauli in itertools.product((q0, q1), Pauli.XYZ):
        eq.make_equality_group(
            lambda: PauliString({q: pauli}, False),
            lambda: PauliString.from_single(q, pauli))
        eq.make_equality_group(
            lambda: PauliString({q: pauli}, True),
            lambda: PauliString.from_single(q, pauli).inverse())
    for q, p0, p1 in itertools.product((q0, q1), Pauli.XYZ, Pauli.XYZ):
        eq.make_equality_group(lambda: PauliString({q: p0, q2: p1}, False))


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_getitem(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = PauliString(qubit_pauli_map)
    for key in qubit_pauli_map:
        assert qubit_pauli_map[key] == pauli_string[key]
    with pytest.raises(KeyError):
        _ = qubit_pauli_map[other]
    with pytest.raises(KeyError):
        _ = pauli_string[other]


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_get(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = PauliString(qubit_pauli_map)
    for key in qubit_pauli_map:
        assert qubit_pauli_map.get(key) == pauli_string.get(key)
    assert qubit_pauli_map.get(other) == pauli_string.get(other) == None
    assert qubit_pauli_map.get(other, 5) == pauli_string.get(other, 5) == 5


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_contains(qubit_pauli_map):
    other = cirq.NamedQubit('other')
    pauli_string = PauliString(qubit_pauli_map)
    for key in qubit_pauli_map:
        assert key in pauli_string
    assert other not in pauli_string


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_keys(qubit_pauli_map):
    pauli_string = PauliString(qubit_pauli_map)
    assert (len(qubit_pauli_map.keys()) == len(pauli_string.keys())
            == len(pauli_string.qubits()))
    assert (set(qubit_pauli_map.keys()) == set(pauli_string.keys())
            == set(pauli_string.qubits()))


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_items(qubit_pauli_map):
    pauli_string = PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map.items()) == len(pauli_string.items())
    assert set(qubit_pauli_map.items()) == set(pauli_string.items())


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_values(qubit_pauli_map):
    pauli_string = PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map.values()) == len(pauli_string.values())
    assert set(qubit_pauli_map.values()) == set(pauli_string.values())


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_len(qubit_pauli_map):
    pauli_string = PauliString(qubit_pauli_map)
    assert len(qubit_pauli_map) == len(pauli_string)


@pytest.mark.parametrize('qubit_pauli_map', _sample_qubit_pauli_maps())
def test_iter(qubit_pauli_map):
    pauli_string = PauliString(qubit_pauli_map)
    assert len(tuple(qubit_pauli_map)) == len(tuple(pauli_string))
    assert set(tuple(qubit_pauli_map)) == set(tuple(pauli_string))


# NamedQubit name repr in Python2 is different: u'q0' vs 'q0'
@cirq.testing.only_test_in_python3
def test_repr():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = PauliString({q2: Pauli.X, q1: Pauli.Y, q0: Pauli.Z})
    assert (repr(pauli_string) ==
            "PauliString({NamedQubit('q0'): Pauli.Z, "
            "NamedQubit('q1'): Pauli.Y, NamedQubit('q2'): Pauli.X}, False)")
    assert (repr(pauli_string.inverse()) ==
            "PauliString({NamedQubit('q0'): Pauli.Z, "
            "NamedQubit('q1'): Pauli.Y, NamedQubit('q2'): Pauli.X}, True)")


def test_str():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = PauliString({q2: Pauli.X, q1: Pauli.Y, q0: Pauli.Z})
    assert str(pauli_string) == '{+, q0:Z, q1:Y, q2:X}'
    assert str(pauli_string.inverse()) == '{-, q0:Z, q1:Y, q2:X}'

@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, {}),
        ({q0: Pauli.X}, {q0: Pauli.Y},
            {q0: (Pauli.X, Pauli.Y)}),
        ({q0: Pauli.X}, {q1: Pauli.X},
            {}),
        ({q0: Pauli.Y, q1: Pauli.Z}, {q1: Pauli.Y, q2: Pauli.X},
            {q1: (Pauli.Z, Pauli.Y)}),
        ({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z}, {},
            {}),
        ({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z}, {q0: Pauli.Y, q1: Pauli.Z},
            {q0: (Pauli.X, Pauli.Y), q1: (Pauli.Y, Pauli.Z)}),
    ))(*_make_qubits(3)))
def test_zip_items(map1, map2, out):
    ps1 = PauliString(map1)
    ps2 = PauliString(map2)
    out_actual = tuple(ps1.zip_items(ps2))
    assert len(out_actual) == len(out)
    assert dict(out_actual) == out


@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, ()),
        ({q0: Pauli.X}, {q0: Pauli.Y},
            ((Pauli.X, Pauli.Y),)),
        ({q0: Pauli.X}, {q1: Pauli.X},
            ()),
        ({q0: Pauli.Y, q1: Pauli.Z}, {q1: Pauli.Y, q2: Pauli.X},
            ((Pauli.Z, Pauli.Y),)),
        ({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z}, {},
            ()),
        ({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z}, {q0: Pauli.Y, q1: Pauli.Z},
            ((Pauli.X, Pauli.Y), (Pauli.Y, Pauli.Z)))  # Order not necessary
    ))(*_make_qubits(3)))
def test_zip_paulis(map1, map2, out):
    ps1 = PauliString(map1)
    ps2 = PauliString(map2)
    out_actual = tuple(ps1.zip_paulis(ps2))
    assert len(out_actual) == len(out)
    if len(out) <= 1:
        assert out_actual == out
    assert set(out_actual) == set(out)  # Ignore output order


def test_commutes_with_string():
    q0, q1, q2 = _make_qubits(3)

    assert PauliString.from_single(q0, Pauli.X).commutes_with_string(
           PauliString.from_single(q0, Pauli.X))
    assert not PauliString.from_single(q0, Pauli.X).commutes_with_string(
               PauliString.from_single(q0, Pauli.Y))
    assert PauliString.from_single(q0, Pauli.X).commutes_with_string(
           PauliString.from_single(q1, Pauli.X))
    assert PauliString.from_single(q0, Pauli.X).commutes_with_string(
           PauliString.from_single(q1, Pauli.Y))

    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.X, q1: Pauli.Y}))
    assert not PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
               PauliString({q0: Pauli.X, q1: Pauli.Z}))
    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.Y, q1: Pauli.X}))
    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.Y, q1: Pauli.Z}))

    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.X, q1: Pauli.Y, q2: Pauli.Z}))
    assert not PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
               PauliString({q0: Pauli.X, q1: Pauli.Z, q2: Pauli.Z}))
    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.Y, q1: Pauli.X, q2: Pauli.Z}))
    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q0: Pauli.Y, q1: Pauli.Z, q2: Pauli.X}))

    assert PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
           PauliString({q2: Pauli.X, q1: Pauli.Y}))
    assert not PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
               PauliString({q2: Pauli.X, q1: Pauli.Z}))
    assert not PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
               PauliString({q2: Pauli.Y, q1: Pauli.X}))
    assert not PauliString({q0: Pauli.X, q1: Pauli.Y}).commutes_with_string(
               PauliString({q2: Pauli.Y, q1: Pauli.Z}))


def test_inverse():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: Pauli.X, q1: Pauli.Y}
    ps1 = PauliString(qubit_pauli_map)
    ps2 = PauliString(qubit_pauli_map, True)
    assert ps1.inverse() == ps2
    assert ps1 == ps2.inverse()
    assert ps1.inverse().inverse() == ps1


def test_map_qubits():
    a, b = (cirq.NamedQubit(name) for name in 'ab')
    q0, q1 = _make_qubits(2)
    qubit_pauli_map1 = {a: Pauli.X, b: Pauli.Y}
    qubit_pauli_map2 = {q0: Pauli.X, q1: Pauli.Y}
    qubit_map = {a: q0, b: q1}
    ps1 = PauliString(qubit_pauli_map1)
    ps2 = PauliString(qubit_pauli_map2)
    assert ps1.map_qubits(qubit_map) == ps2


def _assert_pass_over(ops, before, after):
    assert before.pass_operations_over(ops) == after
    assert (after.pass_operations_over(ops, after_to_before=True)
            == before)


@pytest.mark.parametrize('shift,t_or_f',
        itertools.product(range(3), (True, False)))
def test_pass_operations_over_single(shift, t_or_f):
    q0, q1 = _make_qubits(2)
    X, Y, Z = (pauli+shift for pauli in Pauli.XYZ)

    op0 = CliffordGate.from_pauli(Y)(q1)
    ps_before = PauliString({q0: X}, t_or_f)
    ps_after = ps_before
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = CliffordGate.from_pauli(X)(q0)
    op1 = CliffordGate.from_pauli(Y)(q1)
    ps_before = PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = ps_before
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = CliffordGate.from_double_map({Z: (X,False), X: (Z,False)})(q0)
    ps_before = PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = PauliString({q0: Z, q1: Y}, t_or_f)
    _assert_pass_over([op0], ps_before, ps_after)

    op1 = CliffordGate.from_pauli(X)(q1)
    ps_before = PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = ps_before.inverse()
    _assert_pass_over([op1], ps_before, ps_after)

    ps_after = PauliString({q0: Z, q1: Y}, not t_or_f)
    _assert_pass_over([op0, op1], ps_before, ps_after)


@pytest.mark.parametrize('shift,t_or_f1, t_or_f2,inv',
        itertools.product(range(3), *((True, False),)*3))
def test_pass_operations_over_double(shift, t_or_f1, t_or_f2, inv):
    q0, q1, q2 = _make_qubits(3)
    X, Y, Z = (pauli+shift for pauli in Pauli.XYZ)

    op0 = PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = PauliString({q0: Z, q2: Y}, inv)
    ps_after = PauliString({q0: Z, q2: Y}, inv)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = PauliString({q0: Z, q2: Y}, inv)
    ps_after = PauliString({q0: Z, q2: Y, q1: X}, inv)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = PauliString({q0: Z, q1: Y}, inv)
    ps_after = PauliString({q1: Y}, inv)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = PauliString({q0: Z, q1: Y}, inv)
    ps_after = PauliString({q0: X, q1: Z}, inv ^ t_or_f1 ^ t_or_f2)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = PauliInteractionGate(X, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = PauliString({q0: Z, q1: Y}, inv)
    ps_after = PauliString({q0: Y, q1: Z}, not inv ^ t_or_f1 ^ t_or_f2)
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_unsupported_operations_over():
    q0, = _make_qubits(1)
    pauli_string = PauliString({q0: Pauli.X})
    with pytest.raises(TypeError):
        pauli_string.pass_operations_over([cirq.X(q0)])
