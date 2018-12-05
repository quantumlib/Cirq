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
import numpy as np
import pytest
from cirq.testing import (
    EqualsTester,
)

import cirq


def _make_qubits(n):
    return [cirq.NamedQubit('q{}'.format(i)) for i in range(n)]

def _sample_qubit_pauli_maps():
    qubits = _make_qubits(3)
    paulis_or_none = (None,) + cirq.Pauli.XYZ
    for paulis in itertools.product(paulis_or_none, repeat=len(qubits)):
        yield {qubit: pauli for qubit, pauli in zip(qubits, paulis)
                            if pauli is not None}


def test_eq_ne_hash():
    q0, q1, q2 = _make_qubits(3)
    eq = EqualsTester()
    eq.make_equality_group(
        lambda: cirq.PauliString({}),
        lambda: cirq.PauliString({}, False))
    eq.add_equality_group(cirq.PauliString({}, True))
    for q, pauli in itertools.product((q0, q1), cirq.Pauli.XYZ):
        eq.add_equality_group(cirq.PauliString({q: pauli}, False))
        eq.add_equality_group(cirq.PauliString({q: pauli}, True))
    for q, p0, p1 in itertools.product((q0, q1), cirq.Pauli.XYZ,
                                       cirq.Pauli.XYZ):
        eq.add_equality_group(cirq.PauliString({q: p0, q2: p1}, False))


def test_equal_up_to_sign():
    q0, = _make_qubits(1)
    assert cirq.PauliString({}, False).equal_up_to_sign(
           cirq.PauliString({}, False))
    assert cirq.PauliString({}, True).equal_up_to_sign(
           cirq.PauliString({}, True))
    assert cirq.PauliString({}, False).equal_up_to_sign(
           cirq.PauliString({}, True))

    assert cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
           cirq.PauliString({q0: cirq.Pauli.X}, False))
    assert cirq.PauliString({q0: cirq.Pauli.X}, True).equal_up_to_sign(
           cirq.PauliString({q0: cirq.Pauli.X}, True))
    assert cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
           cirq.PauliString({q0: cirq.Pauli.X}, True))

    assert not cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
               cirq.PauliString({q0: cirq.Pauli.Y}, False))
    assert not cirq.PauliString({q0: cirq.Pauli.X}, True).equal_up_to_sign(
               cirq.PauliString({q0: cirq.Pauli.Y}, True))
    assert not cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
               cirq.PauliString({q0: cirq.Pauli.Y}, True))

    assert not cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
               cirq.PauliString({}, False))
    assert not cirq.PauliString({q0: cirq.Pauli.X}, True).equal_up_to_sign(
               cirq.PauliString({}, True))
    assert not cirq.PauliString({q0: cirq.Pauli.X}, False).equal_up_to_sign(
               cirq.PauliString({}, True))



@pytest.mark.parametrize('pauli', cirq.Pauli.XYZ)
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


# NamedQubit name repr in Python2 is different: u'q0' vs 'q0'
@cirq.testing.only_test_in_python3
def test_repr():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = cirq.PauliString({q2: cirq.Pauli.X, q1: cirq.Pauli.Y,
                                     q0: cirq.Pauli.Z})
    assert (repr(pauli_string) ==
            "cirq.PauliString({cirq.NamedQubit('q0'): cirq.Pauli.Z, "
            "cirq.NamedQubit('q1'): cirq.Pauli.Y, cirq.NamedQubit('q2'): "
            "cirq.Pauli.X}, False)")
    assert (repr(pauli_string.negate()) ==
            "cirq.PauliString({cirq.NamedQubit('q0'): cirq.Pauli.Z, "
            "cirq.NamedQubit('q1'): cirq.Pauli.Y, cirq.NamedQubit('q2'): "
            "cirq.Pauli.X}, True)")


def test_str():
    q0, q1, q2 = _make_qubits(3)
    pauli_string = cirq.PauliString({q2: cirq.Pauli.X, q1: cirq.Pauli.Y,
                                     q0: cirq.Pauli.Z})
    assert str(pauli_string) == '{+, q0:Z, q1:Y, q2:X}'
    assert str(pauli_string.negate()) == '{-, q0:Z, q1:Y, q2:X}'


@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, {}),
        ({q0: cirq.Pauli.X}, {q0: cirq.Pauli.Y},
            {q0: (cirq.Pauli.X, cirq.Pauli.Y)}),
        ({q0: cirq.Pauli.X}, {q1: cirq.Pauli.X},
            {}),
        ({q0: cirq.Pauli.Y, q1: cirq.Pauli.Z},
            {q1: cirq.Pauli.Y, q2: cirq.Pauli.X},
            {q1: (cirq.Pauli.Z, cirq.Pauli.Y)}),
        ({q0: cirq.Pauli.X, q1: cirq.Pauli.Y, q2: cirq.Pauli.Z}, {},
            {}),
        ({q0: cirq.Pauli.X, q1: cirq.Pauli.Y, q2: cirq.Pauli.Z},
            {q0: cirq.Pauli.Y, q1: cirq.Pauli.Z},
            {q0: (cirq.Pauli.X, cirq.Pauli.Y),
             q1: (cirq.Pauli.Y, cirq.Pauli.Z)}),
    ))(*_make_qubits(3)))
def test_zip_items(map1, map2, out):
    ps1 = cirq.PauliString(map1)
    ps2 = cirq.PauliString(map2)
    out_actual = tuple(ps1.zip_items(ps2))
    assert len(out_actual) == len(out)
    assert dict(out_actual) == out


@pytest.mark.parametrize('map1,map2,out', (lambda q0, q1, q2: (
        ({}, {}, ()),
        ({q0: cirq.Pauli.X}, {q0: cirq.Pauli.Y},
            ((cirq.Pauli.X, cirq.Pauli.Y),)),
        ({q0: cirq.Pauli.X}, {q1: cirq.Pauli.X},
            ()),
        ({q0: cirq.Pauli.Y, q1: cirq.Pauli.Z},
            {q1: cirq.Pauli.Y, q2: cirq.Pauli.X},
            ((cirq.Pauli.Z, cirq.Pauli.Y),)),
        ({q0: cirq.Pauli.X, q1: cirq.Pauli.Y, q2: cirq.Pauli.Z}, {},
            ()),
        ({q0: cirq.Pauli.X, q1: cirq.Pauli.Y, q2: cirq.Pauli.Z},
            {q0: cirq.Pauli.Y, q1: cirq.Pauli.Z},
            # Order not necessary
            ((cirq.Pauli.X, cirq.Pauli.Y), (cirq.Pauli.Y, cirq.Pauli.Z)))
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

    assert cirq.PauliString.from_single(q0, cirq.Pauli.X).commutes_with(
           cirq.PauliString.from_single(q0, cirq.Pauli.X))
    assert not cirq.PauliString.from_single(q0, cirq.Pauli.X).commutes_with(
               cirq.PauliString.from_single(q0, cirq.Pauli.Y))
    assert cirq.PauliString.from_single(q0, cirq.Pauli.X).commutes_with(
           cirq.PauliString.from_single(q1, cirq.Pauli.X))
    assert cirq.PauliString.from_single(q0, cirq.Pauli.X).commutes_with(
           cirq.PauliString.from_single(q1, cirq.Pauli.Y))

    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}))
    assert not cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
                                ).commutes_with(
               cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Z}))
    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.Y, q1: cirq.Pauli.X}))
    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.Y, q1: cirq.Pauli.Z}))

    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y,
                             q2: cirq.Pauli.Z}))
    assert not cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
                                ).commutes_with(
               cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Z,
                            q2: cirq.Pauli.Z}))
    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.Y, q1: cirq.Pauli.X,
                             q2: cirq.Pauli.Z}))
    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q0: cirq.Pauli.Y, q1: cirq.Pauli.Z,
                             q2: cirq.Pauli.X}))

    assert cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}).commutes_with(
           cirq.PauliString({q2: cirq.Pauli.X, q1: cirq.Pauli.Y}))
    assert not cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
                                ).commutes_with(
               cirq.PauliString({q2: cirq.Pauli.X, q1: cirq.Pauli.Z}))
    assert not cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
                                ).commutes_with(
               cirq.PauliString({q2: cirq.Pauli.Y, q1: cirq.Pauli.X}))
    assert not cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
                                ).commutes_with(
               cirq.PauliString({q2: cirq.Pauli.Y, q1: cirq.Pauli.Z}))


def test_negate():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    ps2 = cirq.PauliString(qubit_pauli_map, True)
    assert ps1.negate() == -ps1 == ps2
    assert ps1 == ps2.negate() == -ps2
    assert ps1.negate().negate() == ps1


def test_pos():
    q0, q1 = _make_qubits(2)
    qubit_pauli_map = {q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
    ps1 = cirq.PauliString(qubit_pauli_map)
    assert ps1 == +ps1


def test_map_qubits():
    a, b = (cirq.NamedQubit(name) for name in 'ab')
    q0, q1 = _make_qubits(2)
    qubit_pauli_map1 = {a: cirq.Pauli.X, b: cirq.Pauli.Y}
    qubit_pauli_map2 = {q0: cirq.Pauli.X, q1: cirq.Pauli.Y}
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
    pauli_string = cirq.PauliString({q0: cirq.Pauli.X, q1: cirq.Pauli.X,
                                     q2: cirq.Pauli.Y, q3: cirq.Pauli.Y,
                                     q4: cirq.Pauli.Z, q5: cirq.Pauli.Z})
    circuit = cirq.Circuit.from_ops(
                    pauli_string.to_z_basis_ops())

    initial_state = cirq.kron(x0, x1, y0, y1, z0, z1)
    z_basis_state = circuit.apply_unitary_effect_to_state(initial_state)

    expected_state = np.zeros(2 ** 6)
    expected_state[0b010101] = 1

    cirq.testing.assert_allclose_up_to_global_phase(
                    z_basis_state, expected_state, rtol=1e-7, atol=1e-7)


def _assert_pass_over(ops, before, after):
    assert before.pass_operations_over(ops[::-1]) == after
    assert (after.pass_operations_over(ops, after_to_before=True)
            == before)


@pytest.mark.parametrize('shift,t_or_f',
        itertools.product(range(3), (True, False)))
def test_pass_operations_over_single(shift, t_or_f):
    q0, q1 = _make_qubits(2)
    X, Y, Z = (pauli+shift for pauli in cirq.Pauli.XYZ)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X}, t_or_f)
    ps_after = ps_before
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(X)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(Y)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = ps_before
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_double_map({Z: (X,False),
                                                        X: (Z,False)})(q0)
    ps_before = cirq.PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = cirq.PauliString({q0: Z, q1: Y}, t_or_f)
    _assert_pass_over([op0], ps_before, ps_after)

    op1 = cirq.SingleQubitCliffordGate.from_pauli(X)(q1)
    ps_before = cirq.PauliString({q0: X, q1: Y}, t_or_f)
    ps_after = ps_before.negate()
    _assert_pass_over([op1], ps_before, ps_after)

    ps_after = cirq.PauliString({q0: Z, q1: Y}, not t_or_f)
    _assert_pass_over([op0, op1], ps_before, ps_after)

    op0 = cirq.SingleQubitCliffordGate.from_pauli(Z, True)(q0)
    op1 = cirq.SingleQubitCliffordGate.from_pauli(X, True)(q0)
    ps_before = cirq.PauliString({q0: X}, t_or_f)
    ps_after = cirq.PauliString({q0: Y}, not t_or_f)
    _assert_pass_over([op0, op1], ps_before, ps_after)


@pytest.mark.parametrize('shift,t_or_f1, t_or_f2,neg',
        itertools.product(range(3), *((True, False),)*3))
def test_pass_operations_over_double(shift, t_or_f1, t_or_f2, neg):
    q0, q1, q2 = _make_qubits(3)
    X, Y, Z = (pauli+shift for pauli in cirq.Pauli.XYZ)

    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q2: Y}, neg)
    ps_after = cirq.PauliString({q0: Z, q2: Y}, neg)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q2: Y}, neg)
    ps_after = cirq.PauliString({q0: Z, q2: Y, q1: X}, neg)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Z, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, neg)
    ps_after = cirq.PauliString({q1: Y}, neg)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(Y, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, neg)
    ps_after = cirq.PauliString({q0: X, q1: Z}, neg ^ t_or_f1 ^ t_or_f2)
    _assert_pass_over([op0], ps_before, ps_after)

    op0 = cirq.PauliInteractionGate(X, t_or_f1, X, t_or_f2)(q0, q1)
    ps_before = cirq.PauliString({q0: Z, q1: Y}, neg)
    ps_after = cirq.PauliString({q0: Y, q1: Z}, not neg ^ t_or_f1 ^ t_or_f2)
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_operations_over_cz():
    q0, q1 = _make_qubits(2)
    op0 = cirq.CZ(q0, q1)
    ps_before = cirq.PauliString({q0: cirq.Pauli.Z, q1: cirq.Pauli.Y})
    ps_after = cirq.PauliString({q1: cirq.Pauli.Y})
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_operations_over_no_common_qubits():
    class DummyGate(cirq.Gate):
        pass

    q0, q1 = _make_qubits(2)
    op0 = DummyGate()(q1)
    ps_before = cirq.PauliString({q0: cirq.Pauli.Z})
    ps_after = cirq.PauliString({q0: cirq.Pauli.Z})
    _assert_pass_over([op0], ps_before, ps_after)


def test_pass_unsupported_operations_over():
    q0, = _make_qubits(1)
    pauli_string = cirq.PauliString({q0: cirq.Pauli.X})
    with pytest.raises(TypeError):
        pauli_string.pass_operations_over([cirq.X(q0)])


def test_with_qubits():
    old_qubits = cirq.LineQubit.range(9)
    new_qubits = cirq.LineQubit.range(9, 18)
    qubit_pauli_map = {q: cirq.Pauli.XYZ[q.x % 3] for q in old_qubits}
    pauli_string = cirq.PauliString(qubit_pauli_map, negated=True)
    new_pauli_string = pauli_string.with_qubits(*new_qubits)

    assert new_pauli_string.qubits == tuple(new_qubits)
    for q in new_qubits:
        assert new_pauli_string[q] == cirq.Pauli.XYZ[q.x % 3]
    assert new_pauli_string.negated is True
