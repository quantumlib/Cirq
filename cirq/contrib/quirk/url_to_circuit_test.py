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
import json
from typing import Dict, List, Sequence, Tuple, Optional

import numpy as np
import pytest
import sympy

import cirq
from cirq.contrib.quirk.url_to_circuit import quirk_url_to_circuit


def assert_url_to_circuit_returns(
        json_text: str,
        expected_circuit: 'cirq.Circuit' = None,
        *,
        unitary: Optional[np.ndarray] = None,
        diagram: Optional[str] = None,
        output_amplitudes_from_quirk: Optional[List[Dict[str, float]]] = None,
        maps: Optional[Dict[int, int]] = None):
    """
    Args:
        json_text: The part of the quirk URL after "#circuit=".
        expected_circuit: The optional expected circuit. If specified and not
            equal to the parsed circuit, an assertion fails.
        unitary: The optional expected unitary of the circuit. If specified
            and the parsed circuit has a different unitary, an assertion fails.
        diagram: The optional expected circuit diagram. If specified and the
            parsed circuit has a different diagram, an assertion fails.
        output_amplitudes_from_quirk: Optional data copied from Quirk's "export
            simulation data" function, for comparison to Cirq's simulator
            results. If specified and the output from the simulation differs
            from this data (after accounting for differences in endian-ness),
            an assertion fails.
        maps: Optional dictionary of test computational basis input states and
            the output computational basis state that they should be mapped to.
            If any state is mapped to the wrong thing, an assertion fails. Note
            that the states are specified using Quirk's little endian
            convention, meaning that the last bit of a binary literal will refer
            to the last qubit's value instead of vice versa.
    """
    circuit = quirk_url_to_circuit(
        f'https://algassert.com/quirk#circuit={json_text}')

    if diagram is not None:
        cirq.testing.assert_has_diagram(circuit, diagram)

    if expected_circuit is not None:
        cirq.testing.assert_same_circuits(circuit, expected_circuit)

    if unitary is not None:
        np.testing.assert_allclose(cirq.unitary(circuit), unitary, atol=1e-8)

    if output_amplitudes_from_quirk is not None:
        expected = np.array([
            float(e['r']) + 1j * float(e['i'])
            for e in output_amplitudes_from_quirk
        ])

        np.testing.assert_allclose(
            cirq.final_wavefunction(
                circuit,
                # Match Quirk's endian-ness for comparison purposes.
                qubit_order=sorted(circuit.all_qubits(), reverse=True),
            ),
            expected,
            atol=1e-8)

    if maps:
        keys = sorted(maps.keys())
        amps = [
            np.exp(1j * i / len(maps)) / len(maps)**0.5
            for i in range(len(maps))
        ]

        n = len(circuit.all_qubits())
        input_state = np.zeros(1 << n, dtype=np.complex128)
        for k, amp in zip(keys, amps):
            input_state[k] = amp

        output_state = cirq.final_wavefunction(circuit,
                                               initial_state=input_state)

        actual_map = {}
        for k, amp in zip(keys, amps):
            for i, amp2 in enumerate(output_state):
                if abs(amp2 - amp) < 1e-5:
                    actual_map[k] = i

        for k in keys:
            assert actual_map[k] == maps[k], (
                f'{_bin_dec(k)} was mapped to '
                f'{_bin_dec(actual_map.get(k))} '
                f'instead of {_bin_dec(maps[k])}.')


def _bin_dec(x: Optional[int]) -> str:
    if x is None:
        return 'None'
    return f'{bin(x)} ({x})'


def test_parse_simple_cases():
    a, b = cirq.LineQubit.range(2)

    assert quirk_url_to_circuit('http://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#') == cirq.Circuit()
    assert quirk_url_to_circuit(
        'http://algassert.com/quirk#circuit={"cols":[]}') == cirq.Circuit()

    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={'
        '%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22]]'
        '}') == cirq.Circuit(cirq.H(a),
                             cirq.X(b).controlled_by(a))


def test_parse_failures():
    with pytest.raises(ValueError, match='must start with'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#bad')
    with pytest.raises(json.JSONDecodeError):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit=')
    with pytest.raises(ValueError, match='top-level dictionary'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit=[]')
    with pytest.raises(ValueError, match='"cols" entry'):
        _ = quirk_url_to_circuit('http://algassert.com/quirk#circuit={}')
    with pytest.raises(ValueError, match='cols must be a list'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": 1}')
    with pytest.raises(ValueError, match='col must be a list'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [0]}')
    with pytest.raises(ValueError, match='Unrecognized'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [[0]]}')
    with pytest.raises(ValueError, match='Unrecognized'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={"cols": [["not a real"]]}')


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


def test_assert_url_to_circuit_returns():
    a, b = cirq.LineQubit.range(2)

    assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}',
                                  cirq.Circuit(
                                      cirq.X(a),
                                      cirq.X(b),
                                      cirq.X(a),
                                  ),
                                  output_amplitudes_from_quirk=[{
                                      "r": 0,
                                      "i": 0
                                  }, {
                                      "r": 0,
                                      "i": 0
                                  }, {
                                      "r": 1,
                                      "i": 0
                                  }, {
                                      "r": 0,
                                      "i": 0
                                  }])

    assert_url_to_circuit_returns(
        '{"cols":[["X","X"],["X"]]}',
        cirq.Circuit(
            cirq.X(a),
            cirq.X(b),
            cirq.X(a),
        ))

    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns('{"cols":[["X","X"],["X"]]}',
                                      cirq.Circuit(
                                          cirq.X(a),
                                          cirq.X(b),
                                          cirq.X(a),
                                      ),
                                      output_amplitudes_from_quirk=[{
                                          "r": 0,
                                          "i": 0
                                      }, {
                                          "r": 0,
                                          "i": -1
                                      }, {
                                          "r": 0,
                                          "i": 0
                                      }, {
                                          "r": 0,
                                          "i": 0
                                      }])

    with pytest.raises(AssertionError, match='differs from expected circuit'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","X"],["X"]]}',
            cirq.Circuit(
                cirq.X(a),
                cirq.Y(b),
                cirq.X(a),
            ))


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

    assert_url_to_circuit_returns('{"cols":[["…"]]}',
                                  cirq.Circuit())

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


def test_fixed_single_qubit_rotations():
    a, b, c, d = cirq.LineQubit.range(4)

    assert_url_to_circuit_returns(
        '{"cols":[["H","X","Y","Z"]]}',
        cirq.Circuit(cirq.H(a), cirq.X(b), cirq.Y(c), cirq.Z(d)))

    assert_url_to_circuit_returns(
        '{"cols":[["X^½","X^⅓","X^¼"],'
        '["X^⅛","X^⅟₁₆","X^⅟₃₂"],'
        '["X^-½","X^-⅓","X^-¼"],'
        '["X^-⅛","X^-⅟₁₆","X^-⅟₃₂"]]}',
        cirq.Circuit(
            cirq.X(a)**(1 / 2),
            cirq.X(b)**(1 / 3),
            cirq.X(c)**(1 / 4),
            cirq.X(a)**(1 / 8),
            cirq.X(b)**(1 / 16),
            cirq.X(c)**(1 / 32),
            cirq.X(a)**(-1 / 2),
            cirq.X(b)**(-1 / 3),
            cirq.X(c)**(-1 / 4),
            cirq.X(a)**(-1 / 8),
            cirq.X(b)**(-1 / 16),
            cirq.X(c)**(-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Y^½","Y^⅓","Y^¼"],'
        '["Y^⅛","Y^⅟₁₆","Y^⅟₃₂"],'
        '["Y^-½","Y^-⅓","Y^-¼"],'
        '["Y^-⅛","Y^-⅟₁₆","Y^-⅟₃₂"]]}',
        cirq.Circuit(
            cirq.Y(a)**(1 / 2),
            cirq.Y(b)**(1 / 3),
            cirq.Y(c)**(1 / 4),
            cirq.Y(a)**(1 / 8),
            cirq.Y(b)**(1 / 16),
            cirq.Y(c)**(1 / 32),
            cirq.Y(a)**(-1 / 2),
            cirq.Y(b)**(-1 / 3),
            cirq.Y(c)**(-1 / 4),
            cirq.Y(a)**(-1 / 8),
            cirq.Y(b)**(-1 / 16),
            cirq.Y(c)**(-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Z^½","Z^⅓","Z^¼"],'
        '["Z^⅛","Z^⅟₁₆","Z^⅟₃₂"],'
        '["Z^⅟₆₄","Z^⅟₁₂₈"],'
        '["Z^-½","Z^-⅓","Z^-¼"],'
        '["Z^-⅛","Z^-⅟₁₆"]]}',
        cirq.Circuit(
            cirq.Z(a)**(1 / 2),
            cirq.Z(b)**(1 / 3),
            cirq.Z(c)**(1 / 4),
            cirq.Z(a)**(1 / 8),
            cirq.Z(b)**(1 / 16),
            cirq.Z(c)**(1 / 32),
            cirq.Z(a)**(1 / 64),
            cirq.Z(b)**(1 / 128),
            cirq.Z(a)**(-1 / 2),
            cirq.Z(b)**(-1 / 3),
            cirq.Z(c)**(-1 / 4),
            cirq.Z(a)**(-1 / 8),
            cirq.Z(b)**(-1 / 16),
        ))


def test_dynamic_single_qubit_rotations():
    a, b, c = cirq.LineQubit.range(3)
    t = sympy.Symbol('t')

    # Dynamic single qubit rotations.
    assert_url_to_circuit_returns(
        '{"cols":[["X^t","Y^t","Z^t"],["X^-t","Y^-t","Z^-t"]]}',
        cirq.Circuit(
            cirq.X(a)**t,
            cirq.Y(b)**t,
            cirq.Z(c)**t,
            cirq.X(a)**-t,
            cirq.Y(b)**-t,
            cirq.Z(c)**-t,
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["e^iXt","e^iYt","e^iZt"],["e^-iXt","e^-iYt","e^-iZt"]]}',
        cirq.Circuit(
            cirq.Rx(2 * sympy.pi * t).on(a),
            cirq.Ry(2 * sympy.pi * t).on(b),
            cirq.Rz(2 * sympy.pi * t).on(c),
            cirq.Rx(2 * sympy.pi * -t).on(a),
            cirq.Ry(2 * sympy.pi * -t).on(b),
            cirq.Rz(2 * sympy.pi * -t).on(c),
        ))


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
            cirq.measure(a, key='row=0,col=0'),
            cirq.measure(b, key='row=1,col=0'),
            cirq.measure(c, key='row=2,col=0'),
            cirq.Y(a)**-0.5,
            cirq.X(b)**0.5,
        ))


def test_formulaic_gates():
    a, b = cirq.LineQubit.range(2)
    t = sympy.Symbol('t')

    assert_url_to_circuit_returns(
        '{"cols":[["X^ft",{"id":"X^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.X(a)**sympy.sin(sympy.pi * t),
            cirq.X(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Y^ft",{"id":"Y^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Y(a)**sympy.sin(sympy.pi * t),
            cirq.Y(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Z^ft",{"id":"Z^ft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Z(a)**sympy.sin(sympy.pi * t),
            cirq.Z(b)**(t * t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rxft",{"id":"Rxft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Rx(sympy.pi * t * t).on(a),
            cirq.Rx(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Ryft",{"id":"Ryft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Ry(sympy.pi * t * t).on(a),
            cirq.Ry(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rzft",{"id":"Rzft","arg":"t*t"}]]}',
        cirq.Circuit(
            cirq.Rz(sympy.pi * t * t).on(a),
            cirq.Rz(t * t).on(b),
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


def test_arithmetic_comparison_gates():
    with pytest.raises(ValueError, match='Missing input'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":'
                                 '[["^A<B"]]}')
    assert_url_to_circuit_returns('{"cols":[["^A<B","inputA2",1,"inputB2"]]}',
                                  diagram="""
0: ───Quirk(^A<B)───
      │
1: ───A0────────────
      │
2: ───A1────────────
      │
3: ───B0────────────
      │
4: ───B1────────────
        """,
                                  maps={
                                      0b_0_00_10: 0b_1_00_10,
                                      0b_1_00_10: 0b_0_00_10,
                                      0b_0_11_10: 0b_0_11_10,
                                      0b_0_10_10: 0b_0_10_10,
                                      0b_0_01_10: 0b_1_01_10,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^A>B","inputA2",1,"inputB2"]]}',
                                  maps={
                                      0b_0_11_10: 0b_1_11_10,
                                      0b_0_10_10: 0b_0_10_10,
                                      0b_0_01_10: 0b_0_01_10,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^A>=B","inputA2",1,"inputB2"]]}',
                                  maps={
                                      0b_0_11_10: 0b_1_11_10,
                                      0b_0_10_10: 0b_1_10_10,
                                      0b_0_01_10: 0b_0_01_10,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^A<=B","inputA2",1,"inputB2"]]}',
                                  maps={
                                      0b_0_11_10: 0b_0_11_10,
                                      0b_0_10_10: 0b_1_10_10,
                                      0b_0_01_10: 0b_1_01_10,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^A=B","inputA2",1,"inputB2"]]}',
                                  maps={
                                      0b_0_11_10: 0b_0_11_10,
                                      0b_0_10_10: 0b_1_10_10,
                                      0b_0_01_10: 0b_0_01_10,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^A!=B","inputA2",1,"inputB2"]]}',
                                  maps={
                                      0b_0_11_10: 0b_1_11_10,
                                      0b_0_10_10: 0b_0_10_10,
                                      0b_0_01_10: 0b_1_01_10,
                                  })


def test_arithmetic_unlisted_misc_gates():
    assert_url_to_circuit_returns('{"cols":[["^=A3",1,1,"inputA2"]]}',
                                  maps={
                                      0b_000_00: 0b_000_00,
                                      0b_000_01: 0b_001_01,
                                      0b_000_10: 0b_010_10,
                                      0b_111_11: 0b_100_11,
                                  })

    assert_url_to_circuit_returns('{"cols":[["^=A2",1,"inputA3"]]}',
                                  maps={
                                      0b_00_000: 0b_00_000,
                                      0b_00_001: 0b_01_001,
                                      0b_00_010: 0b_10_010,
                                      0b_00_100: 0b_00_100,
                                      0b_11_111: 0b_00_111,
                                  })

    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":5}],["^=A4"]]}',
                                  maps={
                                      0b_0000: 0b_0101,
                                      0b_1111: 0b_1010,
                                  })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":11}],["+cntA4"]]}',
        maps={
            0b_0000: 0b_0011,
            0b_0001: 0b_0100,
            0b_1111: 0b_0010,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5}],["+cntA4"]]}',
        maps={
            0b_0000: 0b_0010,
            0b_0001: 0b_0011,
            0b_1111: 0b_0001,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":7}],["-cntA4"]]}',
        maps={
            0b_0000: 0b_1101,
            0b_0001: 0b_1110,
            0b_1111: 0b_1100,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5}],["Flip<A4"]]}',
        maps={
            0b_1111: 0b_1111,
            0b_0110: 0b_0110,
            0b_0101: 0b_0101,
            0b_0100: 0b_0000,
            0b_0011: 0b_0001,
            0b_0010: 0b_0010,
            0b_0001: 0b_0011,
            0b_0000: 0b_0100,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":6}],["Flip<A4"]]}',
        maps={
            0b_1111: 0b_1111,
            0b_0110: 0b_0110,
            0b_0101: 0b_0000,
            0b_0100: 0b_0001,
            0b_0011: 0b_0010,
            0b_0010: 0b_0011,
            0b_0001: 0b_0100,
            0b_0000: 0b_0101,
        })


def test_arithmetic_addition_gates():
    assert_url_to_circuit_returns('{"cols":[["inc3"]]}',
                                  diagram="""
0: ───Quirk(inc3)───
      │
1: ───#2────────────
      │
2: ───#3────────────
            """,
                                  maps={
                                      0: 1,
                                      3: 4,
                                      7: 0,
                                  })
    assert_url_to_circuit_returns('{"cols":[["dec3"]]}',
                                  maps={
                                      0: 7,
                                      3: 2,
                                      7: 6,
                                  })
    assert_url_to_circuit_returns('{"cols":[["+=A2",1,"inputA2"]]}',
                                  maps={
                                      0b_00_00: 0b_00_00,
                                      0b_01_10: 0b_11_10,
                                      0b_10_11: 0b_01_11,
                                  })
    assert_url_to_circuit_returns('{"cols":[["-=A2",1,"inputA2"]]}',
                                  maps={
                                      0b_00_00: 0b_00_00,
                                      0b_01_10: 0b_11_10,
                                      0b_10_11: 0b_11_11,
                                  })


def test_arithmetic_multiply_accumulate_gates():
    assert_url_to_circuit_returns('{"cols":[["+=AA4",1,1,1,"inputA2"]]}',
                                  maps={
                                      0b_0000_00: 0b_0000_00,
                                      0b_0100_10: 0b_1000_10,
                                      0b_1000_11: 0b_0001_11,
                                  })

    assert_url_to_circuit_returns('{"cols":[["-=AA4",1,1,1,"inputA2"]]}',
                                  maps={
                                      0b_0000_00: 0b_0000_00,
                                      0b_0100_10: 0b_0000_10,
                                      0b_1000_11: 0b_1111_11,
                                  })

    assert_url_to_circuit_returns(
        '{"cols":[["+=AB3",1,1,"inputA2",1,"inputB2"]]}',
        maps={
            0b_000_00_00: 0b_000_00_00,
            0b_000_11_10: 0b_110_11_10,
            0b_100_11_11: 0b_101_11_11,
        })

    assert_url_to_circuit_returns(
        '{"cols":[["-=AB3",1,1,"inputA2",1,"inputB2"]]}',
        maps={
            0b_000_00_00: 0b_000_00_00,
            0b_000_11_10: 0b_010_11_10,
            0b_100_11_11: 0b_011_11_11,
        })


def test_modular_arithmetic_modulus_size():
    with pytest.raises(ValueError, match='too small for modulus'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '[{"id":"setR","arg":17}],["incmodR4"]]}')

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":16}],["incmodR4"]]}')
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":15}],["incmodR4"]]}')

    with pytest.raises(ValueError, match='too small for modulus'):
        _ = quirk_url_to_circuit('https://algassert.com/quirk#circuit={"cols":['
                                 '["incmodR2",1,"inputR3"]]}')

    assert_url_to_circuit_returns('{"cols":[["incmodR3",1,1,"inputR3"]]}')

    assert_url_to_circuit_returns('{"cols":[["incmodR4",1,1,1,"inputR3"]]}')

    assert_url_to_circuit_returns('{"cols":[["incmodR2",1,"inputR2"]]}')


def test_arithmetic_modular_addition_gates():
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":16}],["incmodR4"]]}',
        diagram="""
0: ───Quirk(incmodR4,r=16)───
      │
1: ───#2─────────────────────
      │
2: ───#3─────────────────────
      │
3: ───#4─────────────────────
        """,
        maps={
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            15: 0,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":5}],["incmodR4"]]}',
        maps={
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 0,
            5: 5,
            15: 15,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":5}],["decmodR4"]]}',
        maps={
            0: 4,
            1: 0,
            2: 1,
            3: 2,
            4: 3,
            5: 5,
            15: 15,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3}],["+AmodR4"]]}',
        maps={
            0: 3,
            1: 4,
            2: 0,
            3: 1,
            4: 2,
            5: 5,
            15: 15,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3}],["-AmodR4"]]}',
        maps={
            0: 2,
            1: 3,
            2: 4,
            3: 0,
            4: 1,
            5: 5,
            15: 15,
        })


def test_arithmetic_modular_multiply_accumulate_gates():
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":5},{"id":"setA","arg":3},'
        '{"id":"setB","arg":4}],["+ABmodR4"]]}',
        maps={
            0: 2,
            1: 3,
            2: 4,
            3: 0,
            4: 1,
            5: 5,
            15: 15,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setR","arg":27},{"id":"setA","arg":3},'
        '{"id":"setB","arg":5}],["-ABmodR6"]]}',
        maps={
            0: 27 - 15,
            1: 27 - 14,
            15: 0,
            16: 1,
            26: 26 - 15,
            27: 27,
            63: 63,
        })


def test_arithmetic_multiply_gates():
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":3}],["*A4"]]}',
                                  maps={
                                      0: 0,
                                      1: 3,
                                      3: 9,
                                      9: 11,
                                      11: 1,
                                  })
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":3}],["/A4"]]}',
                                  maps={
                                      0: 0,
                                      1: 11,
                                      3: 1,
                                      9: 3,
                                      11: 9,
                                  })

    # Irreversible multipliers have no effect.
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":4}],["*A4"]]}',
                                  maps={
                                      0: 0,
                                      1: 1,
                                      3: 3,
                                  })
    assert_url_to_circuit_returns('{"cols":[[{"id":"setA","arg":4}],["/A4"]]}',
                                  maps={
                                      0: 0,
                                      1: 1,
                                      3: 3,
                                  })


def test_arithmetic_modular_multiply_gates():
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":3},{"id":"setR","arg":7}],["*AmodR4"]]}',
        maps={
            0: 0,
            1: 3,
            3: 2,
            2: 6,
            6: 4,
            4: 5,
            5: 1,
            7: 7,
            15: 15,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":3},{"id":"setR","arg":7}],["/AmodR4"]]}',
        maps={
            0: 0,
            1: 5,
            2: 3,
            3: 1,
            4: 6,
            5: 4,
            6: 2,
            7: 7,
            15: 15,
        })

    # Irreversible multipliers have no effect.
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5},{"id":"setR","arg":15}],["*AmodR4"]]}',
        maps={
            0: 0,
            1: 1,
            3: 3,
            15: 15,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5},{"id":"setR","arg":15}],["/AmodR4"]]}',
        maps={
            0: 0,
            1: 1,
            3: 3,
            15: 15,
        })


def test_arithmetic_modular_exponentiation_gates():
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5},{"id":"setB","arg":3},'
        '{"id":"setR","arg":7}],["*BToAmodR4"]]}',
        maps={
            0: 0,
            1: 5,
            2: 3,
            15: 15,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":6},{"id":"setB","arg":3},'
        '{"id":"setR","arg":7}],["*BToAmodR4"]]}',
        maps={
            0: 0,
            1: 1,
            2: 2,
            15: 15,
        })

    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":5},{"id":"setB","arg":3},'
        '{"id":"setR","arg":7}],["/BToAmodR4"]]}',
        maps={
            0: 0,
            1: 3,
            2: 6,
            15: 15,
        })
    assert_url_to_circuit_returns(
        '{"cols":[[{"id":"setA","arg":6},{"id":"setB","arg":3},'
        '{"id":"setR","arg":7}],["/BToAmodR4"]]}',
        maps={
            0: 0,
            1: 1,
            2: 2,
            15: 15,
        })


def test_qubit_permutation_gates():
    assert_url_to_circuit_returns('{"cols":[["X",">>4",1,1,1,"X"]]}',
                                  diagram="""
0: ───X───────────────────

1: ───right_rotate[0>3]───
      │
2: ───right_rotate[1>0]───
      │
3: ───right_rotate[2>1]───
      │
4: ───right_rotate[3>2]───

5: ───X───────────────────
        """,
                                  maps={
                                      0b_000000: 0b_100001,
                                      0b_000010: 0b_100101,
                                      0b_000100: 0b_101001,
                                      0b_001000: 0b_110001,
                                      0b_010000: 0b_100011,
                                      0b_011110: 0b_111111,
                                      0b_010100: 0b_101011,
                                  })
    assert_url_to_circuit_returns('{"cols":[["<<4"]]}',
                                  maps={
                                      0b_0000: 0b_0000,
                                      0b_0001: 0b_1000,
                                      0b_0010: 0b_0001,
                                      0b_0100: 0b_0010,
                                      0b_1000: 0b_0100,
                                      0b_1111: 0b_1111,
                                      0b_1010: 0b_0101,
                                  })

    assert_url_to_circuit_returns('{"cols":[["rev4"]]}',
                                  maps={
                                      0b_0000: 0b_0000,
                                      0b_0001: 0b_1000,
                                      0b_0010: 0b_0100,
                                      0b_0100: 0b_0010,
                                      0b_1000: 0b_0001,
                                      0b_1111: 0b_1111,
                                      0b_1010: 0b_0101,
                                  })
    assert_url_to_circuit_returns('{"cols":[["rev3"]]}',
                                  maps={
                                      0b_000: 0b_000,
                                      0b_001: 0b_100,
                                      0b_010: 0b_010,
                                      0b_100: 0b_001,
                                      0b_111: 0b_111,
                                      0b_101: 0b_101,
                                  })

    assert_url_to_circuit_returns('{"cols":[["weave5"]]}',
                                  maps={
                                      0b_00000: 0b_00000,
                                      0b_00001: 0b_00010,
                                      0b_00010: 0b_01000,
                                      0b_00100: 0b_00001,
                                      0b_01000: 0b_00100,
                                      0b_10000: 0b_10000,
                                      0b_00011: 0b_01010,
                                      0b_11111: 0b_11111,
                                  })
    assert_url_to_circuit_returns('{"cols":[["weave6"]]}',
                                  maps={
                                      0b_000000: 0b_000000,
                                      0b_000001: 0b_000001,
                                      0b_000010: 0b_000100,
                                      0b_000100: 0b_010000,
                                      0b_001000: 0b_000010,
                                      0b_010000: 0b_001000,
                                      0b_100000: 0b_100000,
                                      0b_000111: 0b_010101,
                                      0b_111111: 0b_111111,
                                  })
    assert_url_to_circuit_returns('{"cols":[["split5"]]}',
                                  maps={
                                      0b_00000: 0b_00000,
                                      0b_00001: 0b_00100,
                                      0b_00010: 0b_00001,
                                      0b_00100: 0b_01000,
                                      0b_01000: 0b_00010,
                                      0b_10000: 0b_10000,
                                      0b_01010: 0b_00011,
                                      0b_11111: 0b_11111,
                                  })
    assert_url_to_circuit_returns('{"cols":[["split6"]]}',
                                  maps={
                                      0b_000000: 0b_000000,
                                      0b_000001: 0b_000001,
                                      0b_000010: 0b_001000,
                                      0b_000100: 0b_000010,
                                      0b_001000: 0b_010000,
                                      0b_010000: 0b_000100,
                                      0b_100000: 0b_100000,
                                      0b_010101: 0b_000111,
                                      0b_111111: 0b_111111,
                                  })


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
