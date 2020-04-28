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

import functools
import itertools

import numpy as np
import pytest

import cirq
from cirq.testing import (
    EqualsTester,
    assert_allclose_up_to_global_phase,
)

_bools = (False, True)
_paulis = (cirq.X, cirq.Y, cirq.Z)


def _assert_not_mirror(gate) -> None:
    trans_x = gate.transform(cirq.X)
    trans_y = gate.transform(cirq.Y)
    trans_z = gate.transform(cirq.Z)
    right_handed = (trans_x.flip ^ trans_y.flip ^ trans_z.flip ^
                   (trans_x.to.relative_index(trans_y.to) != 1))
    assert right_handed, 'Mirrors'


def _assert_no_collision(gate) -> None:
    trans_x = gate.transform(cirq.X)
    trans_y = gate.transform(cirq.Y)
    trans_z = gate.transform(cirq.Z)
    assert trans_x.to != trans_y.to, 'Collision'
    assert trans_y.to != trans_z.to, 'Collision'
    assert trans_z.to != trans_x.to, 'Collision'


def _all_rotations():
    for pauli, flip, in itertools.product(_paulis, _bools):
        yield cirq.PauliTransform(pauli, flip)


def _all_rotation_pairs():
    for px, flip_x, pz, flip_z in itertools.product(_paulis, _bools,
                                                    _paulis, _bools):
        if px == pz:
            continue
        yield cirq.PauliTransform(px, flip_x), cirq.PauliTransform(pz, flip_z)


def _all_clifford_gates():
    for trans_x, trans_z in _all_rotation_pairs():
        yield cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)


@pytest.mark.parametrize('pauli,flip_x,flip_z',
    itertools.product(_paulis, _bools, _bools))
def test_init_value_error(pauli, flip_x, flip_z):
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_xz_map((pauli, flip_x),
                                                 (pauli, flip_z))


@pytest.mark.parametrize('trans_x,trans_z', _all_rotation_pairs())
def test_init_from_xz(trans_x, trans_z):
    gate = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
    assert gate.transform(cirq.X) == trans_x
    assert gate.transform(cirq.Z) == trans_z
    _assert_not_mirror(gate)
    _assert_no_collision(gate)


@pytest.mark.parametrize('trans1,trans2,from1',
    ((trans1, trans2, from1)
     for trans1, trans2, from1 in itertools.product(_all_rotations(),
                                                    _all_rotations(),
                                                    _paulis)
     if trans1.to != trans2.to))
def test_init_from_double_map_vs_kwargs(trans1, trans2, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    from1_str, from2_str = (str(frm).lower()+'_to' for frm in (from1, from2))
    gate_kw = cirq.SingleQubitCliffordGate.from_double_map(**{from1_str: trans1,
                                                   from2_str: trans2})
    gate_map = cirq.SingleQubitCliffordGate.from_double_map({from1: trans1,
                                                             from2: trans2})
    # Test initializes the same gate
    assert gate_kw == gate_map

    # Test initializes what was expected
    assert gate_map.transform(from1) == trans1
    assert gate_map.transform(from2) == trans2
    _assert_not_mirror(gate_map)
    _assert_no_collision(gate_map)


@pytest.mark.parametrize(
    'trans1,from1',
    ((trans1, from1)
     for trans1, from1 in itertools.product(_all_rotations(), _paulis)))
def test_init_from_double_invalid(trans1, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    # Test throws on invalid arguments
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({
            from1: trans1,
            from2: trans1
        })


@pytest.mark.parametrize('trans,frm',
    itertools.product(_all_rotations(), _paulis))
def test_init_from_single_map_vs_kwargs(trans, frm):
    from_str = str(frm).lower()+'_to'
    # pylint: disable=unexpected-keyword-arg
    gate_kw = cirq.SingleQubitCliffordGate.from_single_map(**{from_str: trans})
    gate_map = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate_kw == gate_map


@pytest.mark.parametrize('trans,frm',
    ((trans, frm)
     for trans, frm in itertools.product(_all_rotations(), _paulis)
     if trans.to != frm))
def test_init_90rot_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.transform(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to one gate
    assert len(gate.decompose_rotation()) == 1
    # Check that this is a 90 degree rotation gate
    assert (gate.merged_with(gate).merged_with(gate).merged_with(gate)
            == cirq.SingleQubitCliffordGate.I)
    # Check that flipping the transform produces the inverse rotation
    trans_rev = cirq.PauliTransform(trans.to, not trans.flip)
    gate_rev = cirq.SingleQubitCliffordGate.from_single_map({frm: trans_rev})
    assert gate**-1 == gate_rev


@pytest.mark.parametrize('trans,frm',
    ((trans, frm)
     for trans, frm in itertools.product(_all_rotations(), _paulis)
     if trans.to == frm and trans.flip))
def test_init_180rot_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.transform(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to one gate
    assert len(gate.decompose_rotation()) == 1
    # Check that this is a 180 degree rotation gate
    assert gate.merged_with(gate) == cirq.SingleQubitCliffordGate.I


@pytest.mark.parametrize('trans,frm',
    ((trans, frm)
     for trans, frm in itertools.product(_all_rotations(), _paulis)
     if trans.to == frm and not trans.flip))
def test_init_ident_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.transform(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to zero gates
    assert len(gate.decompose_rotation()) == 0
    # Check that this is an identity gate
    assert gate == cirq.SingleQubitCliffordGate.I


@pytest.mark.parametrize('pauli,sqrt,expected', (
        (cirq.X, False, cirq.SingleQubitCliffordGate.X),
        (cirq.Y, False, cirq.SingleQubitCliffordGate.Y),
        (cirq.Z, False, cirq.SingleQubitCliffordGate.Z),
        (cirq.X, True, cirq.SingleQubitCliffordGate.X_sqrt),
        (cirq.Y, True, cirq.SingleQubitCliffordGate.Y_sqrt),
        (cirq.Z, True, cirq.SingleQubitCliffordGate.Z_sqrt)))
def test_init_from_pauli(pauli, sqrt, expected):
    gate = cirq.SingleQubitCliffordGate.from_pauli(pauli, sqrt=sqrt)
    assert gate == expected


def test_pow():
    assert cirq.SingleQubitCliffordGate.X**-1 == cirq.SingleQubitCliffordGate.X
    assert cirq.SingleQubitCliffordGate.H**-1 == cirq.SingleQubitCliffordGate.H
    assert (cirq.SingleQubitCliffordGate.X_sqrt ==
            cirq.SingleQubitCliffordGate.X**0.5)
    assert (cirq.SingleQubitCliffordGate.Y_sqrt ==
            cirq.SingleQubitCliffordGate.Y**0.5)
    assert (cirq.SingleQubitCliffordGate.Z_sqrt ==
            cirq.SingleQubitCliffordGate.Z**0.5)
    assert (cirq.SingleQubitCliffordGate.X_nsqrt ==
            cirq.SingleQubitCliffordGate.X**-0.5)
    assert (cirq.SingleQubitCliffordGate.Y_nsqrt ==
            cirq.SingleQubitCliffordGate.Y**-0.5)
    assert (cirq.SingleQubitCliffordGate.Z_nsqrt ==
            cirq.SingleQubitCliffordGate.Z**-0.5)
    assert (cirq.SingleQubitCliffordGate.X_sqrt**-1 ==
            cirq.SingleQubitCliffordGate.X_nsqrt)
    assert cirq.inverse(cirq.SingleQubitCliffordGate.X_nsqrt) == (
        cirq.SingleQubitCliffordGate.X_sqrt
    )
    with pytest.raises(TypeError):
        _ = cirq.SingleQubitCliffordGate.Z**0.25


def test_init_from_quarter_turns():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 0),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 0),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 0),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 4),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 4),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 4),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 8),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 8),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 8),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, -4),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, -4),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, -4)
    )
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 1),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 5),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 9),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, -3),
    )
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 1),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 5),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, 9),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Y, -3),
    )
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 1),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 5),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, 9),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, -3),
    )
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 2),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 6),
    )
    eq.add_equality_group(
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 3),
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.X, 7),
    )


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_init_from_quarter_turns_reconstruct(gate):
    new_gate = functools.reduce(
                    cirq.SingleQubitCliffordGate.merged_with,
                    (cirq.SingleQubitCliffordGate.from_quarter_turns(pauli, qt)
                     for pauli, qt in gate.decompose_rotation()),
                    cirq.SingleQubitCliffordGate.I)
    assert gate == new_gate


def test_init_invalid():
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map()
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map({})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False)}, y_to=(cirq.Y, False))
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False), cirq.Y: (cirq.Y, False)})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map()
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map(
            {cirq.X: (cirq.X, False)})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map(x_to=(cirq.X, False))
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.Y, False),
             cirq.Y: (cirq.Z, False),
             cirq.Z: (cirq.X, False)})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False),
             cirq.Y: (cirq.X, False)})


def test_eq_ne_and_hash():
    eq = EqualsTester()
    for trans_x, trans_z in _all_rotation_pairs():
        gate_gen = lambda: cirq.SingleQubitCliffordGate.from_xz_map(trans_x,
                                                                    trans_z)
        eq.make_equality_group(gate_gen)


@pytest.mark.parametrize('gate,rep', (
    (cirq.SingleQubitCliffordGate.I,
     'cirq.SingleQubitCliffordGate(X:+X, Y:+Y, Z:+Z)'),
    (cirq.SingleQubitCliffordGate.H,
     'cirq.SingleQubitCliffordGate(X:+Z, Y:-Y, Z:+X)'),
    (cirq.SingleQubitCliffordGate.X,
     'cirq.SingleQubitCliffordGate(X:+X, Y:-Y, Z:-Z)'),
    (cirq.SingleQubitCliffordGate.X_sqrt,
     'cirq.SingleQubitCliffordGate(X:+X, Y:+Z, Z:-Y)')))
def test_repr(gate, rep):
    assert repr(gate) == rep


@pytest.mark.parametrize('gate,trans_y', (
    (cirq.SingleQubitCliffordGate.I,       (cirq.Y, False)),
    (cirq.SingleQubitCliffordGate.H,       (cirq.Y, True)),
    (cirq.SingleQubitCliffordGate.X,       (cirq.Y, True)),
    (cirq.SingleQubitCliffordGate.Y,       (cirq.Y, False)),
    (cirq.SingleQubitCliffordGate.Z,       (cirq.Y, True)),
    (cirq.SingleQubitCliffordGate.X_sqrt,  (cirq.Z, False)),
    (cirq.SingleQubitCliffordGate.X_nsqrt, (cirq.Z, True)),
    (cirq.SingleQubitCliffordGate.Y_sqrt,  (cirq.Y, False)),
    (cirq.SingleQubitCliffordGate.Y_nsqrt, (cirq.Y, False)),
    (cirq.SingleQubitCliffordGate.Z_sqrt,  (cirq.X, True)),
    (cirq.SingleQubitCliffordGate.Z_nsqrt, (cirq.X, False))))
def test_y_rotation(gate, trans_y):
    assert gate.transform(cirq.Y) == trans_y


@pytest.mark.parametrize('gate,gate_equiv', (
    (cirq.SingleQubitCliffordGate.I,       cirq.X ** 0),
    (cirq.SingleQubitCliffordGate.H,       cirq.H),
    (cirq.SingleQubitCliffordGate.X,       cirq.X),
    (cirq.SingleQubitCliffordGate.Y,       cirq.Y),
    (cirq.SingleQubitCliffordGate.Z,       cirq.Z),
    (cirq.SingleQubitCliffordGate.X_sqrt,  cirq.X ** 0.5),
    (cirq.SingleQubitCliffordGate.X_nsqrt, cirq.X ** -0.5),
    (cirq.SingleQubitCliffordGate.Y_sqrt,  cirq.Y ** 0.5),
    (cirq.SingleQubitCliffordGate.Y_nsqrt, cirq.Y ** -0.5),
    (cirq.SingleQubitCliffordGate.Z_sqrt,  cirq.Z ** 0.5),
    (cirq.SingleQubitCliffordGate.Z_nsqrt, cirq.Z ** -0.5)))
def test_decompose(gate, gate_equiv):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0)).unitary()
    mat_check = cirq.Circuit(gate_equiv(q0),).unitary()
    assert_allclose_up_to_global_phase(mat, mat_check, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('gate,gate_equiv', (
    (cirq.SingleQubitCliffordGate.I,       cirq.X ** 0),
    (cirq.SingleQubitCliffordGate.H,       cirq.H),
    (cirq.SingleQubitCliffordGate.X,       cirq.X),
    (cirq.SingleQubitCliffordGate.Y,       cirq.Y),
    (cirq.SingleQubitCliffordGate.Z,       cirq.Z),
    (cirq.SingleQubitCliffordGate.X_sqrt,  cirq.X ** 0.5),
    (cirq.SingleQubitCliffordGate.X_nsqrt, cirq.X ** -0.5),
    (cirq.SingleQubitCliffordGate.Y_sqrt,  cirq.Y ** 0.5),
    (cirq.SingleQubitCliffordGate.Y_nsqrt, cirq.Y ** -0.5),
    (cirq.SingleQubitCliffordGate.Z_sqrt,  cirq.Z ** 0.5),
    (cirq.SingleQubitCliffordGate.Z_nsqrt, cirq.Z ** -0.5)))
def test_known_matrix(gate, gate_equiv):
    assert cirq.has_unitary(gate)
    mat = cirq.unitary(gate)
    mat_check = cirq.unitary(gate_equiv)
    assert_allclose_up_to_global_phase(mat, mat_check, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse(gate):
    assert gate == cirq.inverse(cirq.inverse(gate))


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse_matrix(gate):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0)).unitary()
    mat_inv = cirq.Circuit(cirq.inverse(gate)(q0)).unitary()
    assert_allclose_up_to_global_phase(mat, mat_inv.T.conj(),
                                       rtol=1e-7, atol=1e-7)


def test_commutes_notimplemented_type():
    with pytest.raises(TypeError):
        cirq.commutes(cirq.SingleQubitCliffordGate.X, 'X')
    assert (cirq.commutes(cirq.SingleQubitCliffordGate.X,
                          'X',
                          default='default') == 'default')


@pytest.mark.parametrize('gate,other',
    itertools.product(_all_clifford_gates(),
                      _all_clifford_gates()))
def test_commutes_single_qubit_gate(gate, other):
    q0 = cirq.NamedQubit('q0')
    gate_op = gate(q0)
    other_op = other(q0)
    mat = cirq.Circuit(
        gate_op,
        other_op,
    ).unitary()
    mat_swap = cirq.Circuit(
        other_op,
        gate_op,
    ).unitary()
    commutes = cirq.commutes(gate, other)
    commutes_check = cirq.allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check

    # Test after switching order
    mat_swap = cirq.Circuit(
        gate.equivalent_gate_before(other)(q0),
        gate_op,
    ).unitary()
    assert_allclose_up_to_global_phase(mat, mat_swap, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('gate,pauli,half_turns',
    itertools.product(_all_clifford_gates(),
                      _paulis,
                      (0.1, 0.25, 0.5, -0.5)))
def test_commutes_pauli(gate, pauli, half_turns):
    pauli_gate = pauli ** half_turns
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(
        gate(q0),
        pauli_gate(q0),
    ).unitary()
    mat_swap = cirq.Circuit(
        pauli_gate(q0),
        gate(q0),
    ).unitary()
    commutes = cirq.commutes(gate, pauli)
    commutes_check = cirq.allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check


@pytest.mark.parametrize('gate,sym,exp', (
    (cirq.SingleQubitCliffordGate.I,       'I', 1),
    (cirq.SingleQubitCliffordGate.H,       'H', 1),
    (cirq.SingleQubitCliffordGate.X,       'X', 1),
    (cirq.SingleQubitCliffordGate.X_sqrt,  'X', 0.5),
    (cirq.SingleQubitCliffordGate.X_nsqrt, 'X', -0.5),
    (
        cirq.SingleQubitCliffordGate.from_xz_map(
                (cirq.Y, False), (cirq.X, True)),
        '(X^-0.5-Z^0.5)',
        1
    )))
def test_text_diagram_info(gate, sym, exp):
    assert cirq.circuit_diagram_info(gate) == cirq.CircuitDiagramInfo(
        wire_symbols=(sym,),
        exponent=exp)


def test_from_unitary():

    def _test(clifford_gate):
        u = cirq.unitary(clifford_gate)
        result_gate = cirq.SingleQubitCliffordGate.from_unitary(u)
        assert result_gate == clifford_gate

    _test(cirq.SingleQubitCliffordGate.I)
    _test(cirq.SingleQubitCliffordGate.H)
    _test(cirq.SingleQubitCliffordGate.X)
    _test(cirq.SingleQubitCliffordGate.Y)
    _test(cirq.SingleQubitCliffordGate.Z)
    _test(cirq.SingleQubitCliffordGate.X_nsqrt)


def test_from_untary_with_phase_shift():
    u = np.exp(0.42j) * cirq.unitary(cirq.SingleQubitCliffordGate.Y_sqrt)
    gate = cirq.SingleQubitCliffordGate.from_unitary(u)

    assert gate == cirq.SingleQubitCliffordGate.Y_sqrt


def test_from_unitary_not_clifford():
    # Not a single-qubit gate.
    u = cirq.unitary(cirq.CNOT)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None

    # Not an unitary matrix.
    u = 2 * cirq.unitary(cirq.X)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None

    # Not a Clifford gate.
    u = cirq.unitary(cirq.T)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
