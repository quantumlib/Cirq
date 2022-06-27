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
from scipy import linalg

import cirq


@pytest.mark.parametrize('eigen_gate_type', [cirq.ISwapPowGate, cirq.SwapPowGate])
def test_phase_sensitive_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(eigen_gate_type)


def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(cirq.SWAP(a, b), cirq.SWAP(b, a))
    eq.add_equality_group(cirq.SWAP(a, c))

    eq.add_equality_group(cirq.SWAP(a, b) ** 0.3, cirq.SWAP(b, a) ** 0.3)
    eq.add_equality_group(cirq.SWAP(a, c) ** 0.3)

    eq.add_equality_group(cirq.ISWAP(a, b), cirq.ISWAP(b, a))
    eq.add_equality_group(cirq.ISWAP(a, c))

    eq.add_equality_group(cirq.ISWAP(a, b) ** 0.3, cirq.ISWAP(b, a) ** 0.3)
    eq.add_equality_group(cirq.ISWAP(a, c) ** 0.3)


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit(cirq.SWAP(a, b), cirq.ISWAP(a, b) ** -1)

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ───×───iSwap──────
      │   │
b: ───×───iSwap^-1───
""",
    )

    cirq.testing.assert_has_diagram(
        circuit,
        """
a: ---Swap---iSwap------
      |      |
b: ---Swap---iSwap^-1---
""",
        use_unicode_characters=False,
    )


def test_swap_has_stabilizer_effect():
    assert cirq.has_stabilizer_effect(cirq.SWAP)
    assert cirq.has_stabilizer_effect(cirq.SWAP**2)
    assert not cirq.has_stabilizer_effect(cirq.SWAP**0.5)
    assert not cirq.has_stabilizer_effect(cirq.SWAP ** sympy.Symbol('foo'))


def test_swap_unitary():
    # yapf: disable
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.SWAP**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ]))
    # yapf: enable


def test_iswap_unitary():
    # yapf: disable
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.ISWAP),
        # Reference for the iswap gate's matrix using +i instead of -i:
        # https://quantumcomputing.stackexchange.com/questions/2594/
        np.array([[1, 0, 0, 0],
                   [0, 0, 1j, 0],
                   [0, 1j, 0, 0],
                   [0, 0, 0, 1]]),
        atol=1e-8)
    # yapf: enable


def test_iswap_inv_unitary():
    # yapf: disable
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.ISWAP_INV),
        # Reference for the iswap gate's matrix using +i instead of -i:
        # https://quantumcomputing.stackexchange.com/questions/2594/
        np.array([[1, 0, 0, 0],
                  [0, 0, -1j, 0],
                  [0, -1j, 0, 0],
                  [0, 0, 0, 1]]),
        atol=1e-8)
    # yapf: enable


def test_sqrt_iswap_unitary():
    # yapf: disable
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.SQRT_ISWAP),
        # Reference for the sqrt-iSWAP gate's matrix:
        # https://arxiv.org/abs/2105.06074
        np.array([[1, 0,         0,         0],
                  [0, 1/2**0.5,  1j/2**0.5, 0],
                  [0, 1j/2**0.5, 1/2**0.5,  0],
                  [0, 0,         0,         1]]),
        atol=1e-8)
    # yapf: enable


def test_sqrt_iswap_inv_unitary():
    # yapf: disable
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.SQRT_ISWAP_INV),
        # Reference for the inv-sqrt-iSWAP gate's matrix:
        # https://arxiv.org/abs/2105.06074
        np.array([[1, 0,          0,          0],
                  [0, 1/2**0.5,   -1j/2**0.5, 0],
                  [0, -1j/2**0.5, 1/2**0.5,   0],
                  [0, 0,          0,          1]]),
        atol=1e-8)
    # yapf: enable


def test_repr():
    assert repr(cirq.SWAP) == 'cirq.SWAP'
    assert repr(cirq.SWAP**0.5) == '(cirq.SWAP**0.5)'

    assert repr(cirq.ISWAP) == 'cirq.ISWAP'
    assert repr(cirq.ISWAP**0.5) == '(cirq.ISWAP**0.5)'

    assert repr(cirq.ISWAP_INV) == 'cirq.ISWAP_INV'
    assert repr(cirq.ISWAP_INV**0.5) == '(cirq.ISWAP**-0.5)'


def test_str():
    assert str(cirq.SWAP) == 'SWAP'
    assert str(cirq.SWAP**0.5) == 'SWAP**0.5'

    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'

    assert str(cirq.ISWAP_INV) == 'ISWAP_INV'
    assert str(cirq.ISWAP_INV**0.5) == 'ISWAP**-0.5'


def test_iswap_decompose_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    decomposed = cirq.Circuit(cirq.decompose_once(cirq.ISWAP(a, b) ** 0.5))
    cirq.testing.assert_has_diagram(
        decomposed,
        """
a: ───@───H───X───T───X───T^-1───H───@───
      │       │       │              │
b: ───X───────@───────@──────────────X───
""",
    )


def test_trace_distance():
    foo = sympy.Symbol('foo')
    sswap = cirq.SWAP**foo
    siswap = cirq.ISWAP**foo
    # These values should have 1.0 or 0.0 directly returned
    assert cirq.trace_distance_bound(sswap) == 1.0
    assert cirq.trace_distance_bound(siswap) == 1.0
    # These values are calculated, so we use approx_eq
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.SWAP**0.3), np.sin(0.3 * np.pi / 2))
    assert cirq.approx_eq(cirq.trace_distance_bound(cirq.ISWAP**0), 0.0)


def test_trace_distance_over_range_of_exponents():
    for exp in np.linspace(0, 4, 20):
        cirq.testing.assert_has_consistent_trace_distance_bound(cirq.SWAP**exp)
        cirq.testing.assert_has_consistent_trace_distance_bound(cirq.ISWAP**exp)


@pytest.mark.parametrize('angle_rads', (-np.pi, -np.pi / 3, -0.1, np.pi / 5))
def test_riswap_unitary(angle_rads):
    actual = cirq.unitary(cirq.riswap(angle_rads))
    c = np.cos(angle_rads)
    s = 1j * np.sin(angle_rads)
    # yapf: disable
    expected = np.array([[1, 0, 0, 0],
                         [0, c, s, 0],
                         [0, s, c, 0],
                         [0, 0, 0, 1]])
    # yapf: enable
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('angle_rads', (-2 * np.pi / 3, -0.2, 0.4, np.pi / 4))
def test_riswap_hamiltonian(angle_rads):
    actual = cirq.unitary(cirq.riswap(angle_rads))
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    xx = np.kron(x, x)
    yy = np.kron(y, y)
    expected = linalg.expm(+0.5j * angle_rads * (xx + yy))
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('angle_rads', (-np.pi / 5, 0.4, 2, np.pi))
def test_riswap_has_consistent_protocols(angle_rads):
    cirq.testing.assert_implements_consistent_protocols(cirq.riswap(angle_rads))
