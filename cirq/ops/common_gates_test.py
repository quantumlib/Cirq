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

import numpy as np
import pytest

import cirq


H = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
HH = cirq.kron(H, H)
QFT2 = np.array([[1, 1, 1, 1],
                 [1, 1j, -1, -1j],
                 [1, -1, 1, -1],
                 [1, -1j, -1, 1j]]) * 0.5


@pytest.mark.parametrize('eigen_gate_type', [
    cirq.CNotPowGate,
    cirq.CZPowGate,
    cirq.HPowGate,
    cirq.ISwapPowGate,
    cirq.SwapPowGate,
    cirq.XPowGate,
    cirq.YPowGate,
    cirq.ZPowGate,
])
def test_eigen_gates_consistent_protocols(eigen_gate_type):
    cirq.testing.assert_eigengate_implements_consistent_protocols(
            eigen_gate_type)


def test_consistent_protocols():
    cirq.testing.assert_implements_consistent_protocols(
            cirq.MeasurementGate(''), qubit_count=1)


def test_cz_init():
    assert cirq.CZPowGate(exponent=0.5).exponent == 0.5
    assert cirq.CZPowGate(exponent=5).exponent == 5
    assert (cirq.CZ**0.5).exponent == 0.5


def test_cz_str():
    assert str(cirq.CZ) == 'CZ'
    assert str(cirq.CZ**0.5) == 'CZ**0.5'
    assert str(cirq.CZ**-0.25) == 'CZ**-0.25'


def test_cz_repr():
    assert repr(cirq.CZ) == 'cirq.CZ'
    assert repr(cirq.CZ**0.5) == '(cirq.CZ**0.5)'
    assert repr(cirq.CZ**-0.25) == '(cirq.CZ**-0.25)'


def test_cz_unitary():
    assert np.allclose(cirq.unitary(cirq.CZ),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1j]]))

    assert np.allclose(cirq.unitary(cirq.CZ**0),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]]))

    assert np.allclose(cirq.unitary(cirq.CZ**-0.5),
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1j]]))


def test_z_init():
    z = cirq.ZPowGate(exponent=5)
    assert z.exponent == 5

    # Canonicalizes exponent for equality, but keeps the inner details.
    assert cirq.Z**0.5 != cirq.Z**-0.5
    assert (cirq.Z**-1)**0.5 == cirq.Z**-0.5
    assert cirq.Z**-1 == cirq.Z


def test_rot_gates_eq():
    eq = cirq.testing.EqualsTester()
    gates = [
        lambda p: cirq.CZ**p,
        lambda p: cirq.X**p,
        lambda p: cirq.Y**p,
        lambda p: cirq.Z**p,
        lambda p: cirq.CNOT**p,
    ]
    for gate in gates:
        eq.add_equality_group(gate(3.5),
                              gate(-0.5))
        eq.make_equality_group(lambda: gate(0))
        eq.make_equality_group(lambda: gate(0.5))

    eq.add_equality_group(cirq.XPowGate(), cirq.XPowGate(exponent=1), cirq.X)
    eq.add_equality_group(cirq.YPowGate(), cirq.YPowGate(exponent=1), cirq.Y)
    eq.add_equality_group(cirq.ZPowGate(), cirq.ZPowGate(exponent=1), cirq.Z)
    eq.add_equality_group(cirq.ZPowGate(exponent=1,
                                        global_shift=-0.5),
                          cirq.ZPowGate(exponent=5,
                                        global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=3,
                                        global_shift=-0.5))
    eq.add_equality_group(cirq.ZPowGate(exponent=1,
                                        global_shift=-0.1))
    eq.add_equality_group(cirq.ZPowGate(exponent=5,
                                        global_shift=-0.1))
    eq.add_equality_group(cirq.CNotPowGate(),
                          cirq.CNotPowGate(exponent=1),
                          cirq.CNOT)
    eq.add_equality_group(cirq.CZPowGate(),
                          cirq.CZPowGate(exponent=1), cirq.CZ)


def test_z_unitary():
    assert np.allclose(cirq.unitary(cirq.Z),
                       np.array([[1, 0], [0, -1]]))
    assert np.allclose(cirq.unitary(cirq.Z**0.5),
                       np.array([[1, 0], [0, 1j]]))
    assert np.allclose(cirq.unitary(cirq.Z**0),
                       np.array([[1, 0], [0, 1]]))
    assert np.allclose(cirq.unitary(cirq.Z**-0.5),
                       np.array([[1, 0], [0, -1j]]))


def test_y_unitary():
    assert np.allclose(cirq.unitary(cirq.Y),
                       np.array([[0, -1j], [1j, 0]]))

    assert np.allclose(cirq.unitary(cirq.Y**0.5),
                       np.array([[1 + 1j, -1 - 1j], [1 + 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.Y**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.Y**-0.5),
                       np.array([[1 - 1j, 1 - 1j], [-1 + 1j, 1 - 1j]]) / 2)


def test_x_unitary():
    assert np.allclose(cirq.unitary(cirq.X),
                       np.array([[0, 1], [1, 0]]))

    assert np.allclose(cirq.unitary(cirq.X**0.5),
                       np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]) / 2)

    assert np.allclose(cirq.unitary(cirq.X**0),
                       np.array([[1, 0], [0, 1]]))

    assert np.allclose(cirq.unitary(cirq.X**-0.5),
                       np.array([[1 - 1j, 1 + 1j], [1 + 1j, 1 - 1j]]) / 2)


def test_h_unitary():
    sqrt = cirq.unitary(cirq.H**0.5)
    m = np.dot(sqrt, sqrt)
    assert np.allclose(m, cirq.unitary(cirq.H), atol=1e-8)


def test_h_init():
    h = cirq.HPowGate(exponent=0.5)
    assert h.exponent == 0.5


def test_h_str():
    assert str(cirq.H) == 'H'
    assert str(cirq.H**0.5) == 'H^0.5'


def test_runtime_types_of_rot_gates():
    for gate_type in [lambda p: cirq.CZPowGate(exponent=p),
                      lambda p: cirq.XPowGate(exponent=p),
                      lambda p: cirq.YPowGate(exponent=p),
                      lambda p: cirq.ZPowGate(exponent=p)]:
        p = gate_type(cirq.Symbol('a'))
        assert cirq.unitary(p, None) is None
        assert cirq.pow(p, 2, None) is None
        assert cirq.inverse(p, None) is None

        c = gate_type(0.5)
        assert cirq.unitary(c, None) is not None
        assert cirq.pow(c, 2) is not None
        assert cirq.inverse(c) is not None


def test_measurement_eq():
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(cirq.MeasurementGate(''),
                          cirq.MeasurementGate('', invert_mask=()))
    eq.add_equality_group(cirq.MeasurementGate('a'))
    eq.add_equality_group(cirq.MeasurementGate('a', invert_mask=(True,)))
    eq.add_equality_group(cirq.MeasurementGate('a', invert_mask=(False,)))
    eq.add_equality_group(cirq.MeasurementGate('b'))


def test_interchangeable_qubit_eq():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')
    eq = cirq.testing.EqualsTester()

    eq.add_equality_group(cirq.SWAP(a, b), cirq.SWAP(b, a))
    eq.add_equality_group(cirq.SWAP(a, c))

    eq.add_equality_group(cirq.CZ(a, b), cirq.CZ(b, a))
    eq.add_equality_group(cirq.CZ(a, c))

    eq.add_equality_group(cirq.CNOT(a, b))
    eq.add_equality_group(cirq.CNOT(b, a))
    eq.add_equality_group(cirq.CNOT(a, c))


def test_text_diagrams():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    circuit = cirq.Circuit.from_ops(
        cirq.SWAP(a, b),
        cirq.X(a),
        cirq.Y(a),
        cirq.Z(a),
        cirq.Z(a)**cirq.Symbol('x'),
        cirq.CZ(a, b),
        cirq.CNOT(a, b),
        cirq.CNOT(b, a),
        cirq.H(a),
        cirq.ISWAP(a, b),
        cirq.ISWAP(a, b)**-1)

    cirq.testing.assert_has_diagram(circuit, """
a: ───×───X───Y───Z───Z^x───@───@───X───H───iSwap───iSwap──────
      │                     │   │   │       │       │
b: ───×─────────────────────@───X───@───────iSwap───iSwap^-1───
""")

    cirq.testing.assert_has_diagram(circuit, """
a: ---swap---X---Y---Z---Z^x---@---@---X---H---iSwap---iSwap------
      |                        |   |   |       |       |
b: ---swap---------------------@---X---@-------iSwap---iSwap^-1---
""", use_unicode_characters=False)


def test_cnot_unitary():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.CNOT**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0.5+0.5j, 0.5-0.5j],
            [0, 0, 0.5-0.5j, 0.5+0.5j],
        ]))


def test_cnot_keyword_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    eq_tester = cirq.testing.EqualsTester()
    eq_tester.add_equality_group(cirq.CNOT(a, b),
                                 cirq.CNOT(control=a, target=b))
    eq_tester.add_equality_group(cirq.CNOT(b, a),
                                 cirq.CNOT(control=b, target=a))


def test_cnot_keyword_not_equal():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(AssertionError):
        eq_tester = cirq.testing.EqualsTester()
        eq_tester.add_equality_group(cirq.CNOT(a, b),
                                     cirq.CNOT(target=a, control=b))


def test_cnot_keyword_too_few_arguments():
    a = cirq.NamedQubit('a')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(control=a)


def test_cnot_mixed_keyword_and_positional_arguments():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(a, target=b)


def test_cnot_unknown_keyword_argument():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    with pytest.raises(ValueError):
        _ = cirq.CNOT(target=a, controlled=b)


def test_cnot_decompose():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    assert cirq.decompose_once(cirq.CNOT(a, b)**cirq.Symbol('x')) is not None


def test_swap_unitary():
    np.testing.assert_almost_equal(
        cirq.unitary(cirq.SWAP**0.5),
        np.array([
            [1, 0, 0, 0],
            [0, 0.5 + 0.5j, 0.5 - 0.5j, 0],
            [0, 0.5 - 0.5j, 0.5 + 0.5j, 0],
            [0, 0, 0, 1]
        ]))


def test_repr():
    assert repr(cirq.X) == 'cirq.X'
    assert repr(cirq.X**0.5) == '(cirq.X**0.5)'

    assert repr(cirq.Z) == 'cirq.Z'
    assert repr(cirq.Z**0.5) == 'cirq.S'
    assert repr(cirq.Z**0.25) == 'cirq.T'
    assert repr(cirq.Z**0.125) == '(cirq.Z**0.125)'

    assert repr(cirq.S) == 'cirq.S'
    assert repr(cirq.S**-1) == '(cirq.S**-1)'
    assert repr(cirq.T) == 'cirq.T'
    assert repr(cirq.T**-1) == '(cirq.T**-1)'

    assert repr(cirq.Y) == 'cirq.Y'
    assert repr(cirq.Y**0.5) == '(cirq.Y**0.5)'

    assert repr(cirq.CNOT) == 'cirq.CNOT'
    assert repr(cirq.CNOT**0.5) == '(cirq.CNOT**0.5)'

    assert repr(cirq.SWAP) == 'cirq.SWAP'
    assert repr(cirq.SWAP ** 0.5) == '(cirq.SWAP**0.5)'

    assert repr(cirq.ISWAP) == 'cirq.ISWAP'
    assert repr(cirq.ISWAP ** 0.5) == '(cirq.ISWAP**0.5)'

    # There should be no floating point error during initialization, and repr
    # should be using the "shortest decimal value closer to X than any other
    # floating point value" strategy, as opposed to the "exactly value in
    # decimal" strategy.
    assert repr(cirq.CZ**0.2) == '(cirq.CZ**0.2)'


def test_str():
    assert str(cirq.X) == 'X'
    assert str(cirq.X**0.5) == 'X**0.5'

    assert str(cirq.Z) == 'Z'
    assert str(cirq.Z**0.5) == 'S'
    assert str(cirq.Z**0.125) == 'Z**0.125'

    assert str(cirq.Y) == 'Y'
    assert str(cirq.Y**0.5) == 'Y**0.5'

    assert str(cirq.CNOT) == 'CNOT'
    assert str(cirq.CNOT**0.5) == 'CNOT**0.5'

    assert str(cirq.SWAP) == 'SWAP'
    assert str(cirq.SWAP**0.5) == 'SWAP**0.5'

    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'


def test_measurement_gate_diagram():
    # Shows key.
    assert cirq.circuit_diagram_info(cirq.MeasurementGate()
                                     ) == cirq.CircuitDiagramInfo(("M('')",))
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(key='test')
    ) == cirq.CircuitDiagramInfo(("M('test')",))

    # Uses known qubit count.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(),
        cirq.CircuitDiagramInfoArgs(
            known_qubits=None,
            known_qubit_count=3,
            use_unicode_characters=True,
            precision=None,
            qubit_map=None
        )) == cirq.CircuitDiagramInfo(("M('')", 'M', 'M'))

    # Shows invert mask.
    assert cirq.circuit_diagram_info(
        cirq.MeasurementGate(invert_mask=(False, True))
    ) == cirq.CircuitDiagramInfo(("M('')", "!M"))

    # Omits key when it is the default.
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b)), """
a: ───M───
      │
b: ───M───
""")
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, invert_mask=(True,))), """
a: ───!M───
      │
b: ───M────
""")
    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(cirq.measure(a, b, key='test')), """
a: ───M('test')───
      │
b: ───M───────────
""")


def test_measure():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    # Empty application.
    with pytest.raises(ValueError, match='empty set of qubits'):
        _ = cirq.measure()

    assert cirq.measure(a) == cirq.MeasurementGate(key='a').on(a)
    assert cirq.measure(a, b) == cirq.MeasurementGate(key='a,b').on(a, b)
    assert cirq.measure(b, a) == cirq.MeasurementGate(key='b,a').on(b, a)
    assert cirq.measure(a, key='b') == cirq.MeasurementGate(key='b').on(a)
    assert cirq.measure(a, invert_mask=(True,)) == cirq.MeasurementGate(
        key='a', invert_mask=(True,)).on(a)

    with pytest.raises(ValueError, match='ndarray'):
        _ = cirq.measure(np.ndarray([1, 0]))

    with pytest.raises(ValueError, match='QubitId'):
        _ = cirq.measure("bork")

def test_measurement_qubit_count_vs_mask_length():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')
    c = cirq.NamedQubit('c')

    _ = cirq.MeasurementGate(invert_mask=(True,)).on(a)
    _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a, b)
    _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b, c)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False)).on(a)
    with pytest.raises(ValueError):
        _ = cirq.MeasurementGate(invert_mask=(True, False, True)).on(a, b)


def test_measure_each():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    assert cirq.measure_each() == []
    assert cirq.measure_each(a) == [cirq.measure(a)]
    assert cirq.measure_each(a, b) == [cirq.measure(a), cirq.measure(b)]

    assert cirq.measure_each(a, b, key_func=lambda e: e.name + '!') == [
        cirq.measure(a, key='a!'),
        cirq.measure(b, key='b!')
    ]


def test_iswap_str():
    assert str(cirq.ISWAP) == 'ISWAP'
    assert str(cirq.ISWAP**0.5) == 'ISWAP**0.5'


def test_iswap_unitary():
    cirq.testing.assert_allclose_up_to_global_phase(
        cirq.unitary(cirq.ISWAP),
        np.array([[1, 0, 0, 0],
                  [0, 0, 1j, 0],
                  [0, 1j, 0, 0],
                  [0, 0, 0, 1]]),
        atol=1e-8)


def test_iswap_decompose_diagram():
    a = cirq.NamedQubit('a')
    b = cirq.NamedQubit('b')

    decomposed = cirq.Circuit.from_ops(
        cirq.decompose_once(cirq.ISWAP(a, b)**0.5))
    cirq.testing.assert_has_diagram(decomposed, """
a: ───@───H───X───T───X───T^-1───H───@───
      │       │       │              │
b: ───X───────@───────@──────────────X───
""")


def test_is_measurement():
    class NotImplementedOperation(cirq.Operation):
        def with_qubits(self, *new_qubits) -> 'NotImplementedOperation':
            raise NotImplementedError()

        @property
        def qubits(self):
            raise NotImplementedError()

    q = cirq.NamedQubit('q')
    assert cirq.MeasurementGate.is_measurement(cirq.measure(q))
    assert cirq.MeasurementGate.is_measurement(cirq.MeasurementGate(key='b'))

    assert not cirq.MeasurementGate.is_measurement(cirq.X(q))
    assert not cirq.MeasurementGate.is_measurement(cirq.X)
    assert not cirq.MeasurementGate.is_measurement(NotImplementedOperation())


def test_rx_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(np.pi / 2)),
        np.array([[s, -s*1j], [-s*1j, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(-np.pi / 2)),
        np.array([[s, s*1j], [s*1j, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(np.pi)),
        np.array([[0, -1j], [-1j, 0]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rx(-np.pi)),
        np.array([[0, 1j], [1j, 0]]))


def test_ry_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(np.pi / 2)),
        np.array([[s, -s], [s, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(-np.pi / 2)),
        np.array([[s, s], [-s, s]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(np.pi)),
        np.array([[0, -1], [1, 0]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Ry(-np.pi)),
        np.array([[0, 1], [-1, 0]]))


def test_rz_unitary():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(np.pi / 2)),
        np.array([[s - s*1j, 0], [0, s + s*1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(-np.pi / 2)),
        np.array([[s + s*1j, 0], [0, s - s*1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(0)),
        np.array([[1, 0], [0, 1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(2 * np.pi)),
        np.array([[-1, 0], [0, -1]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(np.pi)),
        np.array([[-1j, 0], [0, 1j]]))

    np.testing.assert_allclose(
        cirq.unitary(cirq.Rz(-np.pi)),
        np.array([[1j, 0], [0, -1j]]))


def test_phase_by_xy():
    assert cirq.phase_by(cirq.X, 0.25, 0) == cirq.Y
    assert cirq.phase_by(cirq.Y, 0.25, 0) == cirq.X

    assert cirq.phase_by(cirq.X**0.5, 0.25, 0) == cirq.Y**0.5
    assert cirq.phase_by(cirq.Y**0.5, 0.25, 0) == cirq.X**-0.5
    assert cirq.phase_by(cirq.X**-0.5, 0.25, 0) == cirq.Y**-0.5
    assert cirq.phase_by(cirq.Y**-0.5, 0.25, 0) == cirq.X**0.5


def test_ixyz_circuit_diagram():
    q = cirq.NamedQubit('q')
    ix = cirq.XPowGate(exponent=1, global_shift=0.5)
    iy = cirq.YPowGate(exponent=1, global_shift=0.5)
    iz = cirq.ZPowGate(exponent=1, global_shift=0.5)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            ix(q),
            ix(q)**-1,
            ix(q)**-0.99999,
            ix(q)**-1.00001,
            ix(q)**3,
            ix(q)**4.5,
            ix(q)**4.500001,
        ), """
q: ───X───X───X───X───X───X^0.5───X^0.5───
        """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            iy(q),
            iy(q)**-1,
            iy(q)**3,
            iy(q)**4.5,
            iy(q)**4.500001,
        ), """
q: ───Y───Y───Y───Y^0.5───Y^0.5───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            iz(q),
            iz(q)**-1,
            iz(q)**3,
            iz(q)**4.5,
            iz(q)**4.500001,
        ), """
q: ───Z───Z───Z───S───S───
    """)


def test_rxyz_circuit_diagram():
    q = cirq.NamedQubit('q')

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rx(np.pi).on(q),
            cirq.Rx(-np.pi).on(q),
            cirq.Rx(-np.pi + 0.00001).on(q),
            cirq.Rx(-np.pi - 0.00001).on(q),
            cirq.Rx(3*np.pi).on(q),
            cirq.Rx(7*np.pi/2).on(q),
            cirq.Rx(9*np.pi/2 + 0.00001).on(q),
        ), """
q: ───Rx(π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-π)───Rx(-0.5π)───Rx(0.5π)───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rx(np.pi).on(q),
            cirq.Rx(np.pi/2).on(q),
            cirq.Rx(-np.pi + 0.00001).on(q),
            cirq.Rx(-np.pi - 0.00001).on(q),
        ), """
q: ---Rx(pi)---Rx(0.5pi)---Rx(-pi)---Rx(-pi)---
        """,
        use_unicode_characters=False)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Ry(np.pi).on(q),
            cirq.Ry(-np.pi).on(q),
            cirq.Ry(3 * np.pi).on(q),
            cirq.Ry(9*np.pi/2).on(q),
        ), """
q: ───Ry(π)───Ry(-π)───Ry(-π)───Ry(0.5π)───
    """)

    cirq.testing.assert_has_diagram(
        cirq.Circuit.from_ops(
            cirq.Rz(np.pi).on(q),
            cirq.Rz(-np.pi).on(q),
            cirq.Rz(3 * np.pi).on(q),
            cirq.Rz(9*np.pi/2).on(q),
            cirq.Rz(9*np.pi/2 + 0.00001).on(q),
        ), """
q: ───Rz(π)───Rz(-π)───Rz(-π)───Rz(0.5π)───Rz(0.5π)───
    """)
