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
from typing import Tuple, Type

import numpy as np
import pytest

import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase

_bools = (False, True)
_paulis = (cirq.X, cirq.Y, cirq.Z)


def _assert_not_mirror(gate) -> None:
    trans_x = gate.pauli_tuple(cirq.X)
    trans_y = gate.pauli_tuple(cirq.Y)
    trans_z = gate.pauli_tuple(cirq.Z)
    right_handed = (
        trans_x[1] ^ trans_y[1] ^ trans_z[1] ^ (trans_x[0].relative_index(trans_y[0]) != 1)
    )
    assert right_handed, 'Mirrors'


def _assert_no_collision(gate) -> None:
    trans_x = gate.pauli_tuple(cirq.X)
    trans_y = gate.pauli_tuple(cirq.Y)
    trans_z = gate.pauli_tuple(cirq.Z)
    assert trans_x[0] != trans_y[0], 'Collision'
    assert trans_y[0] != trans_z[0], 'Collision'
    assert trans_z[0] != trans_x[0], 'Collision'


def _all_rotations():
    for pauli, flip in itertools.product(_paulis, _bools):
        yield (pauli, flip)


def _all_rotation_pairs():
    for px, flip_x, pz, flip_z in itertools.product(_paulis, _bools, _paulis, _bools):
        if px == pz:
            continue
        yield (px, flip_x), (pz, flip_z)


@functools.lru_cache()
def _all_clifford_gates() -> Tuple['cirq.SingleQubitCliffordGate', ...]:
    return tuple(
        cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
        for trans_x, trans_z in _all_rotation_pairs()
    )


@pytest.mark.parametrize('pauli,flip_x,flip_z', itertools.product(_paulis, _bools, _bools))
def test_init_value_error(pauli, flip_x, flip_z):
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_xz_map((pauli, flip_x), (pauli, flip_z))


@pytest.mark.parametrize('trans_x,trans_z', _all_rotation_pairs())
def test_init_from_xz(trans_x, trans_z):
    gate = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
    assert gate.pauli_tuple(cirq.X) == trans_x
    assert gate.pauli_tuple(cirq.Z) == trans_z
    _assert_not_mirror(gate)
    _assert_no_collision(gate)


def test_dense_pauli_string():
    gate = cirq.SingleQubitCliffordGate.from_xz_map((cirq.X, True), (cirq.Y, False))
    assert gate.dense_pauli_string(cirq.X) == cirq.DensePauliString('X', coefficient=-1)
    assert gate.dense_pauli_string(cirq.Z) == cirq.DensePauliString('Y')


@pytest.mark.parametrize(
    'trans1,trans2,from1',
    (
        (trans1, trans2, from1)
        for trans1, trans2, from1 in itertools.product(_all_rotations(), _all_rotations(), _paulis)
        if trans1[0] != trans2[0]
    ),
)
def test_init_from_double_map_vs_kwargs(trans1, trans2, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    from1_str, from2_str = (str(frm).lower() + '_to' for frm in (from1, from2))
    gate_kw = cirq.SingleQubitCliffordGate.from_double_map(**{from1_str: trans1, from2_str: trans2})
    gate_map = cirq.SingleQubitCliffordGate.from_double_map({from1: trans1, from2: trans2})
    # Test initializes the same gate
    assert gate_kw == gate_map

    # Test initializes what was expected
    assert gate_map.pauli_tuple(from1) == trans1
    assert gate_map.pauli_tuple(from2) == trans2
    _assert_not_mirror(gate_map)
    _assert_no_collision(gate_map)


@pytest.mark.parametrize(
    'trans1,from1',
    ((trans1, from1) for trans1, from1 in itertools.product(_all_rotations(), _paulis)),
)
def test_init_from_double_invalid(trans1, from1):
    from2 = cirq.Pauli.by_relative_index(from1, 1)
    # Test throws on invalid arguments
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({from1: trans1, from2: trans1})


@pytest.mark.parametrize('trans,frm', itertools.product(_all_rotations(), _paulis))
def test_init_from_single_map_vs_kwargs(trans, frm):
    from_str = str(frm).lower() + '_to'
    # pylint: disable=unexpected-keyword-arg
    gate_kw = cirq.SingleQubitCliffordGate.from_single_map(**{from_str: trans})
    gate_map = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate_kw == gate_map


@pytest.mark.parametrize(
    'trans,frm',
    (
        (trans, frm)
        for trans, frm in itertools.product(_all_rotations(), _paulis)
        if trans[0] != frm
    ),
)
def test_init_90rot_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.pauli_tuple(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to one gate
    assert len(gate.decompose_rotation()) == 1
    # Check that this is a 90 degree rotation gate
    assert (
        gate.merged_with(gate).merged_with(gate).merged_with(gate) == cirq.SingleQubitCliffordGate.I
    )
    # Check that flipping the transform produces the inverse rotation
    trans_rev = (trans[0], not trans[1])
    gate_rev = cirq.SingleQubitCliffordGate.from_single_map({frm: trans_rev})
    assert gate**-1 == gate_rev


@pytest.mark.parametrize(
    'trans,frm',
    (
        (trans, frm)
        for trans, frm in itertools.product(_all_rotations(), _paulis)
        if trans[0] == frm and trans[1]
    ),
)
def test_init_180rot_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.pauli_tuple(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to one gate
    assert len(gate.decompose_rotation()) == 1
    # Check that this is a 180 degree rotation gate
    assert gate.merged_with(gate) == cirq.SingleQubitCliffordGate.I


@pytest.mark.parametrize(
    'trans,frm',
    (
        (trans, frm)
        for trans, frm in itertools.product(_all_rotations(), _paulis)
        if trans[0] == frm and not trans[1]
    ),
)
def test_init_ident_from_single(trans, frm):
    gate = cirq.SingleQubitCliffordGate.from_single_map({frm: trans})
    assert gate.pauli_tuple(frm) == trans
    _assert_not_mirror(gate)
    _assert_no_collision(gate)
    # Check that it decomposes to zero gates
    assert len(gate.decompose_rotation()) == 0
    # Check that this is an identity gate
    assert gate == cirq.SingleQubitCliffordGate.I


@pytest.mark.parametrize(
    'pauli,sqrt,expected',
    (
        (cirq.X, False, cirq.SingleQubitCliffordGate.X),
        (cirq.Y, False, cirq.SingleQubitCliffordGate.Y),
        (cirq.Z, False, cirq.SingleQubitCliffordGate.Z),
        (cirq.X, True, cirq.SingleQubitCliffordGate.X_sqrt),
        (cirq.Y, True, cirq.SingleQubitCliffordGate.Y_sqrt),
        (cirq.Z, True, cirq.SingleQubitCliffordGate.Z_sqrt),
    ),
)
def test_init_from_pauli(pauli, sqrt, expected):
    gate = cirq.SingleQubitCliffordGate.from_pauli(pauli, sqrt=sqrt)
    assert gate == expected


def test_pow():
    assert cirq.SingleQubitCliffordGate.X**-1 == cirq.SingleQubitCliffordGate.X
    assert cirq.SingleQubitCliffordGate.H**-1 == cirq.SingleQubitCliffordGate.H
    assert cirq.SingleQubitCliffordGate.X_sqrt == cirq.SingleQubitCliffordGate.X**0.5
    assert cirq.SingleQubitCliffordGate.Y_sqrt == cirq.SingleQubitCliffordGate.Y**0.5
    assert cirq.SingleQubitCliffordGate.Z_sqrt == cirq.SingleQubitCliffordGate.Z**0.5
    assert cirq.SingleQubitCliffordGate.X_nsqrt == cirq.SingleQubitCliffordGate.X**-0.5
    assert cirq.SingleQubitCliffordGate.Y_nsqrt == cirq.SingleQubitCliffordGate.Y**-0.5
    assert cirq.SingleQubitCliffordGate.Z_nsqrt == cirq.SingleQubitCliffordGate.Z**-0.5
    assert cirq.SingleQubitCliffordGate.X_sqrt**-1 == cirq.SingleQubitCliffordGate.X_nsqrt
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
        cirq.SingleQubitCliffordGate.from_quarter_turns(cirq.Z, -4),
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
        (
            cirq.SingleQubitCliffordGate.from_quarter_turns(pauli, qt)
            for pauli, qt in gate.decompose_rotation()
        ),
        cirq.SingleQubitCliffordGate.I,
    )
    assert gate == new_gate


def test_init_invalid():
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map()
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map({})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False)}, y_to=(cirq.Y, False)
        )
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False), cirq.Y: (cirq.Y, False)}
        )
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map()
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map({cirq.X: (cirq.X, False)})
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_double_map(x_to=(cirq.X, False))
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.Y, False), cirq.Y: (cirq.Z, False), cirq.Z: (cirq.X, False)}
        )
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_single_map(
            {cirq.X: (cirq.X, False), cirq.Y: (cirq.X, False)}
        )


def test_eq_ne_and_hash():
    eq = EqualsTester()
    for trans_x, trans_z in _all_rotation_pairs():
        gate_gen = lambda: cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
        eq.make_equality_group(gate_gen)


@pytest.mark.parametrize(
    'gate',
    (
        cirq.SingleQubitCliffordGate.I,
        cirq.SingleQubitCliffordGate.H,
        cirq.SingleQubitCliffordGate.X,
        cirq.SingleQubitCliffordGate.X_sqrt,
    ),
)
def test_repr_gate(gate):
    cirq.testing.assert_equivalent_repr(gate)


def test_repr_operation():
    cirq.testing.assert_equivalent_repr(
        cirq.SingleQubitCliffordGate.from_pauli(cirq.Z).on(cirq.LineQubit(2))
    )


@pytest.mark.parametrize(
    'gate,trans_y',
    (
        (cirq.SingleQubitCliffordGate.I, (cirq.Y, False)),
        (cirq.SingleQubitCliffordGate.H, (cirq.Y, True)),
        (cirq.SingleQubitCliffordGate.X, (cirq.Y, True)),
        (cirq.SingleQubitCliffordGate.Y, (cirq.Y, False)),
        (cirq.SingleQubitCliffordGate.Z, (cirq.Y, True)),
        (cirq.SingleQubitCliffordGate.X_sqrt, (cirq.Z, False)),
        (cirq.SingleQubitCliffordGate.X_nsqrt, (cirq.Z, True)),
        (cirq.SingleQubitCliffordGate.Y_sqrt, (cirq.Y, False)),
        (cirq.SingleQubitCliffordGate.Y_nsqrt, (cirq.Y, False)),
        (cirq.SingleQubitCliffordGate.Z_sqrt, (cirq.X, True)),
        (cirq.SingleQubitCliffordGate.Z_nsqrt, (cirq.X, False)),
    ),
)
def test_y_rotation(gate, trans_y):
    assert gate.pauli_tuple(cirq.Y) == trans_y


@pytest.mark.parametrize(
    'gate,gate_equiv',
    (
        (cirq.SingleQubitCliffordGate.I, cirq.X**0),
        (cirq.SingleQubitCliffordGate.H, cirq.H),
        (cirq.SingleQubitCliffordGate.X, cirq.X),
        (cirq.SingleQubitCliffordGate.Y, cirq.Y),
        (cirq.SingleQubitCliffordGate.Z, cirq.Z),
        (cirq.SingleQubitCliffordGate.X_sqrt, cirq.X**0.5),
        (cirq.SingleQubitCliffordGate.X_nsqrt, cirq.X**-0.5),
        (cirq.SingleQubitCliffordGate.Y_sqrt, cirq.Y**0.5),
        (cirq.SingleQubitCliffordGate.Y_nsqrt, cirq.Y**-0.5),
        (cirq.SingleQubitCliffordGate.Z_sqrt, cirq.Z**0.5),
        (cirq.SingleQubitCliffordGate.Z_nsqrt, cirq.Z**-0.5),
    ),
)
def test_decompose(gate, gate_equiv):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0)).unitary()
    mat_check = cirq.Circuit(gate_equiv(q0)).unitary()
    assert_allclose_up_to_global_phase(mat, mat_check, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    'gate,gate_equiv',
    (
        (cirq.SingleQubitCliffordGate.I, cirq.X**0),
        (cirq.SingleQubitCliffordGate.H, cirq.H),
        (cirq.SingleQubitCliffordGate.X, cirq.X),
        (cirq.SingleQubitCliffordGate.Y, cirq.Y),
        (cirq.SingleQubitCliffordGate.Z, cirq.Z),
        (cirq.SingleQubitCliffordGate.X_sqrt, cirq.X**0.5),
        (cirq.SingleQubitCliffordGate.X_nsqrt, cirq.X**-0.5),
        (cirq.SingleQubitCliffordGate.Y_sqrt, cirq.Y**0.5),
        (cirq.SingleQubitCliffordGate.Y_nsqrt, cirq.Y**-0.5),
        (cirq.SingleQubitCliffordGate.Z_sqrt, cirq.Z**0.5),
        (cirq.SingleQubitCliffordGate.Z_nsqrt, cirq.Z**-0.5),
    ),
)
def test_known_matrix(gate, gate_equiv):
    assert cirq.has_unitary(gate)
    mat = cirq.unitary(gate)
    mat_check = cirq.unitary(gate_equiv)
    assert_allclose_up_to_global_phase(mat, mat_check, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    'name, expected_cls',
    [
        ('I', cirq.SingleQubitCliffordGate),
        ('H', cirq.SingleQubitCliffordGate),
        ('X', cirq.SingleQubitCliffordGate),
        ('Y', cirq.SingleQubitCliffordGate),
        ('Z', cirq.SingleQubitCliffordGate),
        ('S', cirq.SingleQubitCliffordGate),
        ('X_sqrt', cirq.SingleQubitCliffordGate),
        ('X_nsqrt', cirq.SingleQubitCliffordGate),
        ('Y_sqrt', cirq.SingleQubitCliffordGate),
        ('Y_nsqrt', cirq.SingleQubitCliffordGate),
        ('Z_sqrt', cirq.SingleQubitCliffordGate),
        ('Z_nsqrt', cirq.SingleQubitCliffordGate),
        ('CNOT', cirq.CliffordGate),
        ('CZ', cirq.CliffordGate),
        ('SWAP', cirq.CliffordGate),
    ],
)
def test_common_clifford_types(name: str, expected_cls: Type) -> None:
    assert isinstance(getattr(cirq.CliffordGate, name), expected_cls)
    assert isinstance(getattr(cirq.SingleQubitCliffordGate, name), expected_cls)


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse(gate):
    assert gate == cirq.inverse(cirq.inverse(gate))


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_inverse_matrix(gate):
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0)).unitary()
    mat_inv = cirq.Circuit(cirq.inverse(gate)(q0)).unitary()
    assert_allclose_up_to_global_phase(mat, mat_inv.T.conj(), rtol=1e-7, atol=1e-7)


def test_commutes_notimplemented_type():
    with pytest.raises(TypeError):
        cirq.commutes(cirq.SingleQubitCliffordGate.X, 'X')
    assert cirq.commutes(cirq.SingleQubitCliffordGate.X, 'X', default='default') == 'default'

    with pytest.raises(TypeError):
        cirq.commutes(cirq.CliffordGate.X, 'X')
    assert cirq.commutes(cirq.CliffordGate.X, 'X', default='default') == 'default'


@pytest.mark.parametrize('gate,other', itertools.combinations(_all_clifford_gates(), r=2))
def test_commutes_single_qubit_gate(gate, other):
    q0 = cirq.NamedQubit('q0')
    gate_op = gate(q0)
    other_op = other(q0)
    mat = cirq.Circuit(gate_op, other_op).unitary()
    mat_swap = cirq.Circuit(other_op, gate_op).unitary()
    commutes = cirq.commutes(gate, other)
    commutes_check = cirq.allclose_up_to_global_phase(mat, mat_swap)
    assert commutes == commutes_check

    # Test after switching order
    mat_swap = cirq.Circuit(gate.equivalent_gate_before(other)(q0), gate_op).unitary()
    assert_allclose_up_to_global_phase(mat, mat_swap, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize('gate', _all_clifford_gates())
def test_parses_single_qubit_gate(gate):
    assert gate == cirq.read_json(json_text=(cirq.to_json(gate)))


@pytest.mark.parametrize(
    'gate,pauli,half_turns',
    itertools.product(_all_clifford_gates(), _paulis, (1.0, 0.25, 0.5, -0.5)),
)
def test_commutes_pauli(gate, pauli, half_turns):
    pauli_gate = pauli if half_turns == 1 else pauli**half_turns
    q0 = cirq.NamedQubit('q0')
    mat = cirq.Circuit(gate(q0), pauli_gate(q0)).unitary()
    mat_swap = cirq.Circuit(pauli_gate(q0), gate(q0)).unitary()
    commutes = cirq.commutes(gate, pauli_gate)
    commutes_check = np.allclose(mat, mat_swap)
    assert commutes == commutes_check, f"gate: {gate}, pauli {pauli}"


def test_to_clifford_tableau_util_function():
    tableau = cirq.ops.clifford_gate._to_clifford_tableau(
        x_to=(cirq.X, False), z_to=(cirq.Z, False)
    )
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=0)

    tableau = cirq.ops.clifford_gate._to_clifford_tableau(x_to=(cirq.X, False), z_to=(cirq.Z, True))
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=1)

    tableau = cirq.ops.clifford_gate._to_clifford_tableau(
        rotation_map={cirq.X: (cirq.X, False), cirq.Z: (cirq.Z, False)}
    )
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=0)

    tableau = cirq.ops.clifford_gate._to_clifford_tableau(
        rotation_map={cirq.X: (cirq.X, False), cirq.Z: (cirq.Z, True)}
    )
    assert tableau == cirq.CliffordTableau(num_qubits=1, initial_state=1)

    with pytest.raises(ValueError):
        cirq.ops.clifford_gate._to_clifford_tableau()


@pytest.mark.parametrize(
    'gate,sym,exp',
    (
        (cirq.SingleQubitCliffordGate.I, 'I', 1),
        (cirq.SingleQubitCliffordGate.H, 'H', 1),
        (cirq.SingleQubitCliffordGate.X, 'X', 1),
        (cirq.SingleQubitCliffordGate.X_sqrt, 'X', 0.5),
        (cirq.SingleQubitCliffordGate.X_nsqrt, 'X', -0.5),
        (
            cirq.SingleQubitCliffordGate.from_xz_map((cirq.Y, False), (cirq.X, True)),
            '(X^-0.5-Z^0.5)',
            1,
        ),
    ),
)
def test_text_diagram_info(gate, sym, exp):
    assert cirq.circuit_diagram_info(gate) == cirq.CircuitDiagramInfo(
        wire_symbols=(sym,), exponent=exp
    )


@pytest.mark.parametrize("clifford_gate", cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
def test_from_unitary(clifford_gate):
    u = cirq.unitary(clifford_gate)
    result_gate = cirq.SingleQubitCliffordGate.from_unitary(u)
    assert result_gate == clifford_gate

    result_gate2, global_phase = cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u)
    assert result_gate2 == result_gate
    assert np.allclose(cirq.unitary(result_gate2) * global_phase, u)


def test_from_unitary_with_phase_shift():
    u = np.exp(0.42j) * cirq.unitary(cirq.SingleQubitCliffordGate.Y_sqrt)
    gate = cirq.SingleQubitCliffordGate.from_unitary(u)

    assert gate == cirq.SingleQubitCliffordGate.Y_sqrt

    gate2, global_phase = cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u)
    assert gate2 == gate
    assert np.allclose(cirq.unitary(gate2) * global_phase, u)


def test_from_unitary_not_clifford():
    # Not a single-qubit gate.
    u = cirq.unitary(cirq.CNOT)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None

    # Not an unitary matrix.
    u = 2 * cirq.unitary(cirq.X)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None

    # Not a Clifford gate.
    u = cirq.unitary(cirq.T)
    assert cirq.SingleQubitCliffordGate.from_unitary(u) is None
    assert cirq.SingleQubitCliffordGate.from_unitary_with_global_phase(u) is None


@pytest.mark.parametrize("clifford_gate", cirq.SingleQubitCliffordGate.all_single_qubit_cliffords)
def test_decompose_gate(clifford_gate):
    gates = clifford_gate.decompose_gate()
    u = functools.reduce(np.dot, [np.eye(2), *(cirq.unitary(gate) for gate in reversed(gates))])
    assert np.allclose(u, cirq.unitary(clifford_gate))  # No global phase difference.


@pytest.mark.parametrize('trans_x,trans_z', _all_rotation_pairs())
def test_to_phased_xz_gate(trans_x, trans_z):
    gate = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
    actual_phased_xz_gate = gate.to_phased_xz_gate()._canonical()
    expect_phased_xz_gates = cirq.PhasedXZGate.from_matrix(cirq.unitary(gate))

    assert np.isclose(actual_phased_xz_gate.x_exponent, expect_phased_xz_gates.x_exponent)
    assert np.isclose(actual_phased_xz_gate.z_exponent, expect_phased_xz_gates.z_exponent)
    assert np.isclose(
        actual_phased_xz_gate.axis_phase_exponent, expect_phased_xz_gates.axis_phase_exponent
    )


def test_from_xz_to_clifford_tableau():
    seen_tableau = []
    for trans_x, trans_z in _all_rotation_pairs():
        tableau = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z).clifford_tableau
        tableau_number = sum(2**i * t for i, t in enumerate(tableau.matrix().ravel()))
        tableau_number = tableau_number * 4 + 2 * tableau.rs[0] + tableau.rs[1]
        seen_tableau.append(tableau_number)
        # Satisfy the symplectic property
        assert sum(tableau.matrix()[0, :2] * tableau.matrix()[1, 1::-1]) % 2 == 1

    # Should not have any duplication.
    assert len(set(seen_tableau)) == 24


@pytest.mark.parametrize(
    'clifford_gate,standard_gate',
    [
        (cirq.CliffordGate.I, cirq.I),
        (cirq.CliffordGate.X, cirq.X),
        (cirq.CliffordGate.Y, cirq.Y),
        (cirq.CliffordGate.Z, cirq.Z),
        (cirq.CliffordGate.H, cirq.H),
        (cirq.CliffordGate.S, cirq.S),
        (cirq.CliffordGate.CNOT, cirq.CNOT),
        (cirq.CliffordGate.CZ, cirq.CZ),
        (cirq.CliffordGate.SWAP, cirq.SWAP),
    ],
)
def test_common_clifford_gate(clifford_gate, standard_gate):
    # cirq.unitary is relied on the _decompose_ methods.
    u_c = cirq.unitary(clifford_gate)
    u_s = cirq.unitary(standard_gate)
    cirq.testing.assert_allclose_up_to_global_phase(u_c, u_s, atol=1e-8)


@pytest.mark.parametrize('property_name', ("all_single_qubit_cliffords", "CNOT", "CZ", "SWAP"))
def test_common_clifford_gate_caching(property_name):
    cache_name = f"_{property_name}"
    delattr(cirq.CliffordGate, cache_name)
    assert not hasattr(cirq.CliffordGate, cache_name)
    _ = getattr(cirq.CliffordGate, property_name)
    assert hasattr(cirq.CliffordGate, cache_name)


def test_multi_qubit_clifford_pow():
    assert cirq.CliffordGate.X**-1 == cirq.CliffordGate.X
    assert cirq.CliffordGate.H**-1 == cirq.CliffordGate.H
    assert cirq.CliffordGate.S**2 == cirq.CliffordGate.Z
    assert cirq.CliffordGate.S**-1 == cirq.CliffordGate.S**3
    assert cirq.CliffordGate.S**-3 == cirq.CliffordGate.S
    assert cirq.CliffordGate.CNOT**3 == cirq.CliffordGate.CNOT
    assert cirq.CliffordGate.CNOT**-3 == cirq.CliffordGate.CNOT
    with pytest.raises(TypeError):
        _ = cirq.CliffordGate.Z**0.25


def test_stabilizer_effec():
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.X)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.H)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.S)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.CNOT)
    assert cirq.has_stabilizer_effect(cirq.CliffordGate.CZ)
    qubits = cirq.LineQubit.range(2)
    gate = cirq.CliffordGate.from_op_list(
        [cirq.H(qubits[1]), cirq.CZ(*qubits), cirq.H(qubits[1])], qubits
    )
    assert cirq.has_stabilizer_effect(gate)


def test_clifford_gate_from_op_list():
    # Since from_op_list() ==> _act_on_() ==> tableau.then() and then() has already covered
    # lots of random circuit cases, here we just test a few well-known relationships.
    qubit = cirq.NamedQubit('test')
    gate = cirq.CliffordGate.from_op_list([cirq.X(qubit), cirq.Z(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y  # The tableau ignores the global phase

    gate = cirq.CliffordGate.from_op_list([cirq.Z(qubit), cirq.X(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y  # The tableau ignores the global phase

    gate = cirq.CliffordGate.from_op_list([cirq.X(qubit), cirq.Y(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Z  # The tableau ignores the global phase

    gate = cirq.CliffordGate.from_op_list([cirq.Z(qubit), cirq.X(qubit)], [qubit])
    assert gate == cirq.CliffordGate.Y  # The tableau ignores the global phase

    # Two qubits gates
    qubits = cirq.LineQubit.range(2)
    gate = cirq.CliffordGate.from_op_list(
        [cirq.H(qubits[1]), cirq.CZ(*qubits), cirq.H(qubits[1])], qubits
    )
    assert gate == cirq.CliffordGate.CNOT

    gate = cirq.CliffordGate.from_op_list(
        [cirq.H(qubits[1]), cirq.CNOT(*qubits), cirq.H(qubits[1])], qubits
    )
    assert gate == cirq.CliffordGate.CZ

    # Note the order of qubits matters
    gate = cirq.CliffordGate.from_op_list(
        [cirq.H(qubits[0]), cirq.CZ(qubits[1], qubits[0]), cirq.H(qubits[0])], qubits
    )
    assert gate != cirq.CliffordGate.CNOT
    # But if we reverse the qubit_order, they will equal again.
    gate = cirq.CliffordGate.from_op_list(
        [cirq.H(qubits[0]), cirq.CZ(qubits[1], qubits[0]), cirq.H(qubits[0])], qubits[::-1]
    )
    assert gate == cirq.CliffordGate.CNOT

    with pytest.raises(
        ValueError, match="only be constructed from the operations that has stabilizer effect"
    ):
        cirq.CliffordGate.from_op_list([cirq.T(qubit)], [qubit])


def test_clifford_gate_from_tableau():
    t = cirq.CliffordGate.X.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.X

    t = cirq.CliffordGate.H.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.H

    t = cirq.CliffordGate.CNOT.clifford_tableau
    assert cirq.CliffordGate.from_clifford_tableau(t) == cirq.CliffordGate.CNOT

    with pytest.raises(ValueError, match='Input argument has to be a CliffordTableau instance.'):
        cirq.SingleQubitCliffordGate.from_clifford_tableau(123)

    with pytest.raises(ValueError, match="The number of qubit of input tableau should be 1"):
        t = cirq.CliffordTableau(num_qubits=2)
        cirq.SingleQubitCliffordGate.from_clifford_tableau(t)

    with pytest.raises(ValueError):
        t = cirq.CliffordTableau(num_qubits=1)
        t.xs = np.array([1, 1]).reshape(2, 1)
        t.zs = np.array([1, 1]).reshape(2, 1)  # This violates the sympletic property.
        cirq.CliffordGate.from_clifford_tableau(t)

    with pytest.raises(ValueError, match="Input argument has to be a CliffordTableau instance."):
        cirq.CliffordGate.from_clifford_tableau(1)


def test_multi_clifford_decompose_by_unitary():
    # Construct a random clifford gate:
    n, num_ops = 5, 20  # because we relied on unitary cannot test large-scale qubits
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for _ in range(10):
        qubits = cirq.LineQubit.range(n)
        ops = []
        for _ in range(num_ops):
            g = np.random.randint(len(gate_candidate))
            indices = (np.random.randint(n),) if g < 5 else np.random.choice(n, 2, replace=False)
            ops.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        gate = cirq.CliffordGate.from_op_list(ops, qubits)
        decomposed_ops = cirq.decompose(gate.on(*qubits))
        circ = cirq.Circuit(decomposed_ops)
        circ.append(cirq.I.on_each(qubits))  # make sure the dimension aligned.
        cirq.testing.assert_allclose_up_to_global_phase(
            cirq.unitary(gate), cirq.unitary(circ), atol=1e-7
        )


def test_pad_tableau_bad_input():
    with pytest.raises(
        ValueError, match="Input axes of padding should match with the number of qubits"
    ):
        tableau = cirq.CliffordTableau(num_qubits=3)
        cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=4, axes=[1, 2])

    with pytest.raises(
        ValueError, match='The number of qubits in the input tableau should not be larger than'
    ):
        tableau = cirq.CliffordTableau(num_qubits=3)
        cirq.ops.clifford_gate._pad_tableau(tableau, num_qubits_after_padding=2, axes=[0, 1, 2])


def test_pad_tableau():
    tableau = cirq.CliffordTableau(num_qubits=1)
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(
        tableau, num_qubits_after_padding=2, axes=[0]
    )
    assert padded_tableau == cirq.CliffordTableau(num_qubits=2)

    tableau = cirq.CliffordTableau(num_qubits=1, initial_state=1)
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(
        tableau, num_qubits_after_padding=1, axes=[0]
    )
    assert padded_tableau == cirq.CliffordGate.X.clifford_tableau

    # Tableau for H
    # [0 1 0]
    # [1 0 0]
    tableau = cirq.CliffordGate.H.clifford_tableau
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(
        tableau, num_qubits_after_padding=2, axes=[0]
    )
    # fmt: off
    np.testing.assert_equal(
        padded_tableau.matrix().astype(np.int64),
        np.array([[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]]),
    )
    # fmt: on
    np.testing.assert_equal(padded_tableau.rs.astype(np.int64), np.zeros(4))
    # The tableau of H again but pad for another ax
    tableau = cirq.CliffordGate.H.clifford_tableau
    padded_tableau = cirq.ops.clifford_gate._pad_tableau(
        tableau, num_qubits_after_padding=2, axes=[1]
    )
    # fmt: off
    np.testing.assert_equal(
        padded_tableau.matrix().astype(np.int64),
        np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]]),
    )
    # fmt: on
    np.testing.assert_equal(padded_tableau.rs.astype(np.int64), np.zeros(4))


def test_clifford_gate_act_on_small_case():
    # Note this is also covered by the `from_op_list` one, etc.

    qubits = cirq.LineQubit.range(5)
    args = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=5), qubits=qubits, prng=np.random.RandomState()
    )
    expected_args = cirq.CliffordTableauSimulationState(
        tableau=cirq.CliffordTableau(num_qubits=5), qubits=qubits, prng=np.random.RandomState()
    )
    cirq.act_on(cirq.H, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.H, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau

    cirq.act_on(cirq.CNOT, expected_args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.CNOT, args, qubits=[qubits[0], qubits[1]], allow_decompose=False)
    assert args.tableau == expected_args.tableau

    cirq.act_on(cirq.H, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.H, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau

    cirq.act_on(cirq.S, expected_args, qubits=[qubits[0]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.S, args, qubits=[qubits[0]], allow_decompose=False)
    assert args.tableau == expected_args.tableau

    cirq.act_on(cirq.X, expected_args, qubits=[qubits[2]], allow_decompose=False)
    cirq.act_on(cirq.CliffordGate.X, args, qubits=[qubits[2]], allow_decompose=False)
    assert args.tableau == expected_args.tableau


def test_clifford_gate_act_on_large_case():
    n, num_ops = 50, 1000  # because we don't need unitary, it is fast.
    gate_candidate = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.CNOT, cirq.CZ]
    for seed in range(10):
        prng = np.random.RandomState(seed)
        t1 = cirq.CliffordTableau(num_qubits=n)
        t2 = cirq.CliffordTableau(num_qubits=n)
        qubits = cirq.LineQubit.range(n)
        args1 = cirq.CliffordTableauSimulationState(tableau=t1, qubits=qubits, prng=prng)
        args2 = cirq.CliffordTableauSimulationState(tableau=t2, qubits=qubits, prng=prng)
        ops = []
        for _ in range(0, num_ops, 100):
            g = prng.randint(len(gate_candidate))
            indices = (prng.randint(n),) if g < 5 else prng.choice(n, 2, replace=False)
            cirq.act_on(
                gate_candidate[g], args1, qubits=[qubits[i] for i in indices], allow_decompose=False
            )
            ops.append(gate_candidate[g].on(*[qubits[i] for i in indices]))
        compiled_gate = cirq.CliffordGate.from_op_list(ops, qubits)
        cirq.act_on(compiled_gate, args2, qubits)

        assert args1.tableau == args2.tableau


def test_clifford_gate_act_on_ch_form():
    # Although we don't support CH_form from the _act_on_, it will fall back
    # to the decomposititon method and apply it through decomposed ops.
    # Here we run it for the coverage only.
    args = cirq.StabilizerChFormSimulationState(
        initial_state=cirq.StabilizerStateChForm(num_qubits=2, initial_state=1),
        qubits=cirq.LineQubit.range(2),
        prng=np.random.RandomState(),
    )
    cirq.act_on(cirq.CliffordGate.X, args, qubits=cirq.LineQubit.range(1))
    np.testing.assert_allclose(args.state.state_vector(), np.array([0, 0, 0, 1]))


def test_clifford_gate_act_on_fail():
    with pytest.raises(TypeError, match="Failed to act"):
        cirq.act_on(cirq.CliffordGate.X, ExampleSimulationState(), qubits=())


def test_all_single_qubit_clifford_unitaries():
    i = np.eye(2)
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])

    cs = [cirq.unitary(c) for c in cirq.CliffordGate.all_single_qubit_cliffords]

    # Identity
    assert cirq.equal_up_to_global_phase(cs[0], i)
    # Paulis
    assert cirq.equal_up_to_global_phase(cs[1], x)
    assert cirq.equal_up_to_global_phase(cs[2], y)
    assert cirq.equal_up_to_global_phase(cs[3], z)
    # Square roots of Paulis
    assert cirq.equal_up_to_global_phase(cs[4], (i - 1j * x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[5], (i - 1j * y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[6], (i - 1j * z) / np.sqrt(2))
    # Negative square roots of Paulis
    assert cirq.equal_up_to_global_phase(cs[7], (i + 1j * x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[8], (i + 1j * y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[9], (i + 1j * z) / np.sqrt(2))
    # Hadamards
    assert cirq.equal_up_to_global_phase(cs[10], (z + x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[11], (x + y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[12], (y + z) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[13], (z - x) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[14], (x - y) / np.sqrt(2))
    assert cirq.equal_up_to_global_phase(cs[15], (y - z) / np.sqrt(2))
    # Order-3 Cliffords
    assert cirq.equal_up_to_global_phase(cs[16], (i - 1j * (x + y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[17], (i - 1j * (x + y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[18], (i - 1j * (x - y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[19], (i - 1j * (x - y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[20], (i - 1j * (-x + y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[21], (i - 1j * (-x + y - z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[22], (i - 1j * (-x - y + z)) / 2)
    assert cirq.equal_up_to_global_phase(cs[23], (i - 1j * (-x - y - z)) / 2)
