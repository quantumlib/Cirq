# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import sympy

import cirq
from cirq.contrib.quirk.cells.testing import assert_url_to_circuit_returns
from cirq.contrib.quirk.url_to_circuit import quirk_url_to_circuit


def test_gate_type_swap():
    a, b, c = cirq.LineQubit.range(3)
    assert_url_to_circuit_returns('{"cols":[["Swap","Swap"]]}',
                                  cirq.Circuit(cirq.SWAP(a, b)))
    assert_url_to_circuit_returns('{"cols":[["Swap","X","Swap"]]}',
                                  cirq.Circuit(cirq.SWAP(a, c), cirq.X(b)))

    with pytest.raises(ValueError, match='number of swap gates'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[['
            '"Swap"]]}')
    with pytest.raises(ValueError, match='number of swap gates'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[['
            '"Swap","Swap","Swap"]]}')


def test_gate_type_controls():
    a, b, c = cirq.LineQubit.range(3)

    assert_url_to_circuit_returns('{"cols":[["•","X"]]}',
                                  cirq.Circuit(cirq.X(b).controlled_by(a),))
    assert_url_to_circuit_returns(
        '{"cols":[["◦","X"]]}',
        cirq.Circuit(
            cirq.X(a),
            cirq.X(b).controlled_by(a),
            cirq.X(a),
        ))

    assert_url_to_circuit_returns('{"cols":[["⊕","X"]]}',
                                  cirq.Circuit(
                                      cirq.Y(a)**0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.Y(a)**-0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": -0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                  ])
    assert_url_to_circuit_returns('{"cols":[["⊖","X"]]}',
                                  cirq.Circuit(
                                      cirq.Y(a)**-0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.Y(a)**+0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": -0.5,
                                          "i": 0
                                      },
                                  ])

    assert_url_to_circuit_returns('{"cols":[["⊗","X"]]}',
                                  cirq.Circuit(
                                      cirq.X(a)**-0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.X(a)**+0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": -0.5
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": 0.5
                                      },
                                  ])
    assert_url_to_circuit_returns('{"cols":[["(/)","X"]]}',
                                  cirq.Circuit(
                                      cirq.X(a)**+0.5,
                                      cirq.X(b).controlled_by(a),
                                      cirq.X(a)**-0.5,
                                  ),
                                  output_amplitudes_from_quirk=[
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": 0.5
                                      },
                                      {
                                          "r": 0.5,
                                          "i": 0
                                      },
                                      {
                                          "r": 0,
                                          "i": -0.5
                                      },
                                  ])

    qs = cirq.LineQubit.range(8)
    assert_url_to_circuit_returns(
        '{"cols":[["X","•","◦","⊕","⊖","⊗","(/)","Z"]]}',
        cirq.Circuit(
            cirq.X(qs[2]),
            cirq.Y(qs[3])**0.5,
            cirq.Y(qs[4])**-0.5,
            cirq.X(qs[5])**-0.5,
            cirq.X(qs[6])**0.5,
            cirq.X(qs[0]).controlled_by(*qs[1:7]),
            cirq.Z(qs[7]).controlled_by(*qs[1:7]),
            cirq.X(qs[2]),
            cirq.Y(qs[3])**-0.5,
            cirq.Y(qs[4])**0.5,
            cirq.X(qs[5])**0.5,
            cirq.X(qs[6])**-0.5,
        ))


def test_parity_controls():
    a, b, c, d, e = cirq.LineQubit.range(5)

    assert_url_to_circuit_returns(
        '{"cols":[["Y","xpar","ypar","zpar","Z"]]}',
        cirq.Circuit(
            cirq.Y(b)**0.5,
            cirq.X(c)**-0.5,
            cirq.CNOT(c, b),
            cirq.CNOT(d, b),
            cirq.Y(a).controlled_by(b),
            cirq.Z(e).controlled_by(b),
            cirq.CNOT(d, b),
            cirq.CNOT(c, b),
            cirq.X(c)**0.5,
            cirq.Y(b)**-0.5,
        ))


def test_non_physical_operations():
    with pytest.raises(NotImplementedError, match="Unphysical operation"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["__error__"]]}')
    with pytest.raises(NotImplementedError, match="Unphysical operation"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["__unstable__UniversalNot"]]}')


def test_scalar_operations():
    a = cirq.LineQubit(0)

    assert_url_to_circuit_returns('{"cols":[["…"]]}', cirq.Circuit())

    assert_url_to_circuit_returns('{"cols":[["NeGate"]]}',
                                  cirq.Circuit(cirq.GlobalPhaseOperation(-1)))

    assert_url_to_circuit_returns('{"cols":[["i"]]}',
                                  cirq.Circuit(cirq.GlobalPhaseOperation(1j)))

    assert_url_to_circuit_returns('{"cols":[["-i"]]}',
                                  cirq.Circuit(cirq.GlobalPhaseOperation(-1j)))

    assert_url_to_circuit_returns(
        '{"cols":[["√i"]]}', cirq.Circuit(cirq.GlobalPhaseOperation(1j**0.5)))

    assert_url_to_circuit_returns(
        '{"cols":[["√-i"]]}', cirq.Circuit(cirq.GlobalPhaseOperation(1j**-0.5)))


def test_measurement_gates():
    a, b, c = cirq.LineQubit.range(3)
    assert_url_to_circuit_returns(
        '{"cols":[["Measure","Measure"],["Measure","Measure"]]}',
        cirq.Circuit(
            cirq.measure(a, key='row=0,col=0'),
            cirq.measure(b, key='row=1,col=0'),
            cirq.measure(a, key='row=0,col=1'),
            cirq.measure(b, key='row=1,col=1'),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["XDetector","YDetector","ZDetector"]]}',
        cirq.Circuit(
            cirq.X(b)**-0.5,
            cirq.Y(a)**0.5,
            cirq.Moment([
                cirq.measure(a, key='row=0,col=0'),
                cirq.measure(b, key='row=1,col=0'),
                cirq.measure(c, key='row=2,col=0'),
            ]),
            cirq.Y(a)**-0.5,
            cirq.X(b)**0.5,
        ))


def test_parameterized_single_qubit_rotations():
    with pytest.raises(ValueError, match='classical constant'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":'
                                 '[["Z^(A/2^n)",{"id":"setA","arg":3}]]}')
    with pytest.raises(ValueError, match="Missing input 'a'"):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":'
                                 '[["X^(A/2^n)"]]}')

    assert_url_to_circuit_returns(
        '{"cols":[["Z^(A/2^n)","inputA2"]]}',
        diagram="""
0: ───Z^(A/2^2)───
      │
1: ───A0──────────
      │
2: ───A1──────────
        """,
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns('{"cols":[["Z^(-A/2^n)","inputA1"]]}',
                                  unitary=np.diag([1, 1, 1, -1j]))

    assert_url_to_circuit_returns(
        '{"cols":[["H"],["X^(A/2^n)","inputA2"],["H"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns(
        '{"cols":[["H"],["X^(-A/2^n)","inputA2"],["H"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**-0.5, 1j**-1, 1j**-1.5]))

    assert_url_to_circuit_returns(
        '{"cols":[["X^-½"],["Y^(A/2^n)","inputA2"],["X^½"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**0.5, 1j**1, 1j**1.5]))
    assert_url_to_circuit_returns(
        '{"cols":[["X^-½"],["Y^(-A/2^n)","inputA2"],["X^½"]]}',
        unitary=np.diag([1, 1, 1, 1, 1j**0, 1j**-0.5, 1j**-1, 1j**-1.5]))


def test_frequency_space_gates():
    a, b, c = cirq.LineQubit.range(3)

    assert_url_to_circuit_returns('{"cols":[["QFT3"]]}',
                                  cirq.Circuit(cirq.QFT(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["QFT†3"]]}', cirq.Circuit(cirq.inverse(cirq.QFT(a, b, c)),))

    assert_url_to_circuit_returns(
        '{"cols":[["PhaseGradient3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=0.5)(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["PhaseUngradient3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=-0.5)(a, b, c),))

    t = sympy.Symbol('t')
    assert_url_to_circuit_returns(
        '{"cols":[["grad^t2"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=2, exponent=2 * t)(a, b),))
    assert_url_to_circuit_returns(
        '{"cols":[["grad^t3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=4 * t)(a, b, c),))
    assert_url_to_circuit_returns(
        '{"cols":[["grad^-t3"]]}',
        cirq.Circuit(
            cirq.PhaseGradientGate(num_qubits=3, exponent=-4 * t)(a, b, c),))


def test_displays():
    assert_url_to_circuit_returns(
        '{"cols":[["Amps2"],[1,"Amps3"],["Chance"],'
        '["Chance2"],["Density"],["Density3"],'
        '["Sample4"],["Bloch"],["Sample2"]'
        ']}', cirq.Circuit())


def test_not_implemented_gates():
    # This test mostly exists to ensure the gates are tested if added.

    for k in ["X^⌈t⌉", "X^⌈t-¼⌉", "Counting4", "Uncounting4", ">>t3", "<<t3"]:
        with pytest.raises(NotImplementedError, match="discrete parameter"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')

    for k in ["add3", "sub3", "c+=ab4", "c-=ab4"]:
        with pytest.raises(NotImplementedError, match="deprecated"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')

    for k in ["X", "Y", "Z"]:
        with pytest.raises(NotImplementedError, match="feedback"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"'
                                     'cols":[["' + k + 'DetectControlReset"]]}')

    for k in ["|0⟩⟨0|", "|1⟩⟨1|", "|+⟩⟨+|", "|-⟩⟨-|", "|X⟩⟨X|", "|/⟩⟨/|", "0"]:
        with pytest.raises(NotImplementedError, match="postselection"):
            _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={'
                                     '"cols":[["' + k + '"]]}')


def test_example_qft_circuit():
    qft_example_diagram = """
0: ───×───────────────H───S───────T───────────Z─────────────────────Z────────────────────────────────Z──────────────────────────────────────────Z────────────────────────────────────────────────────Z──────────────────────────────────────────────────────────────
      │                   │       │           │                     │                                │                                          │                                                    │
1: ───┼───×───────────────@───H───┼───S───────┼─────────T───────────┼──────────Z─────────────────────┼─────────Z────────────────────────────────┼─────────Z──────────────────────────────────────────┼─────────Z────────────────────────────────────────────────────
      │   │                       │   │       │         │           │          │                     │         │                                │         │                                          │         │
2: ───┼───┼───×───────────────────@───@───H───┼─────────┼───S───────┼──────────┼─────────T───────────┼─────────┼──────────Z─────────────────────┼─────────┼─────────Z────────────────────────────────┼─────────┼─────────Z──────────────────────────────────────────
      │   │   │                               │         │   │       │          │         │           │         │          │                     │         │         │                                │         │         │
3: ───┼───┼───┼───×───────────────────────────@^(1/8)───@───@───H───┼──────────┼─────────┼───S───────┼─────────┼──────────┼─────────T───────────┼─────────┼─────────┼──────────Z─────────────────────┼─────────┼─────────┼─────────Z────────────────────────────────
      │   │   │   │                                                 │          │         │   │       │         │          │         │           │         │         │          │                     │         │         │         │
4: ───┼───┼───┼───×─────────────────────────────────────────────────@^(1/16)───@^(1/8)───@───@───H───┼─────────┼──────────┼─────────┼───S───────┼─────────┼─────────┼──────────┼─────────T───────────┼─────────┼─────────┼─────────┼──────────Z─────────────────────
      │   │   │                                                                                      │         │          │         │   │       │         │         │          │         │           │         │         │         │          │
5: ───┼───┼───×──────────────────────────────────────────────────────────────────────────────────────@^0.031───@^(1/16)───@^(1/8)───@───@───H───┼─────────┼─────────┼──────────┼─────────┼───S───────┼─────────┼─────────┼─────────┼──────────┼─────────T───────────
      │   │                                                                                                                                     │         │         │          │         │   │       │         │         │         │          │         │
6: ───┼───×─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.016───@^0.031───@^(1/16)───@^(1/8)───@───@───H───┼─────────┼─────────┼─────────┼──────────┼─────────┼───S───────
      │                                                                                                                                                                                              │         │         │         │          │         │   │
7: ───×──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────@^0.008───@^0.016───@^0.031───@^(1/16)───@^(1/8)───@───@───H───
    """

    qft_example_json = (
        '{"cols":['
        # '["Counting8"],'
        '["Chance8"],'
        '["…","…","…","…","…","…","…","…"],'
        '["Swap",1,1,1,1,1,1,"Swap"],'
        '[1,"Swap",1,1,1,1,"Swap"],'
        '[1,1,"Swap",1,1,"Swap"],'
        '[1,1,1,"Swap","Swap"],'
        '["H"],'
        '["Z^½","•"],'
        '[1,"H"],'
        '["Z^¼","Z^½","•"],'
        '[1,1,"H"],'
        '["Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,"H"],'
        '["Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,"H"],'
        '["Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,"H"],'
        '["Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,"H"],'
        '["Z^⅟₁₂₈","Z^⅟₆₄","Z^⅟₃₂","Z^⅟₁₆","Z^⅛","Z^¼","Z^½","•"],'
        '[1,1,1,1,1,1,1,"H"]]}')
    qft_example_json_uri_escaped = (
        '{%22cols%22:['
        # '[%22Counting8%22],'
        '[%22Chance8%22],'
        '[%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,'
        '%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22,%22%E2%80%A6%22],'
        '[%22Swap%22,1,1,1,1,1,1,%22Swap%22],'
        '[1,%22Swap%22,1,1,1,1,%22Swap%22],'
        '[1,1,%22Swap%22,1,1,%22Swap%22],'
        '[1,1,1,%22Swap%22,%22Swap%22],'
        '[%22H%22],'
        '[%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,%22H%22],'
        '[%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,%22H%22],'
        '[%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,%22H%22],'
        '[%22Z^%E2%85%9F%E2%82%81%E2%82%82%E2%82%88%22,'
        '%22Z^%E2%85%9F%E2%82%86%E2%82%84%22,'
        '%22Z^%E2%85%9F%E2%82%83%E2%82%82%22,'
        '%22Z^%E2%85%9F%E2%82%81%E2%82%86%22,%22Z^%E2%85%9B%22,%22Z^%C2%BC%22,'
        '%22Z^%C2%BD%22,%22%E2%80%A2%22],'
        '[1,1,1,1,1,1,1,%22H%22]]}')
    assert_url_to_circuit_returns(qft_example_json, diagram=qft_example_diagram)
    assert_url_to_circuit_returns(qft_example_json_uri_escaped,
                                  diagram=qft_example_diagram)
