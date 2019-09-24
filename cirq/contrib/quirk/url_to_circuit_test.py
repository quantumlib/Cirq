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
from typing import Dict, List

import numpy as np
import pytest
import sympy

import cirq
from cirq.contrib.quirk.url_to_circuit import quirk_url_to_circuit


def test_parse_simple_cases():
    a, b = cirq.LineQubit.range(2)

    assert quirk_url_to_circuit('http://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk') == cirq.Circuit()
    assert quirk_url_to_circuit('https://algassert.com/quirk#'
                                ) == cirq.Circuit()
    assert quirk_url_to_circuit('http://algassert.com/quirk#circuit={"cols":[]}'
                                ) == cirq.Circuit()

    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={'
        '%22cols%22:[[%22H%22],[%22%E2%80%A2%22,%22X%22]]'
        '}'
    ) == cirq.Circuit.from_ops(cirq.H(a), cirq.X(b).controlled_by(a))


def test_parse_failures():
    with pytest.raises(ValueError, match='must start with'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#bad')
    with pytest.raises(json.JSONDecodeError):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit=')
    with pytest.raises(ValueError, match='top-level dictionary'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit=[]')
    with pytest.raises(ValueError, match='"cols" entry'):
        _ = quirk_url_to_circuit(
            'http://algassert.com/quirk#circuit={}')
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
    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={"cols":[['
        '"Swap","Swap"]]}'
    ) == cirq.Circuit.from_ops(cirq.SWAP(a, b))
    assert quirk_url_to_circuit(
        'https://algassert.com/quirk#circuit={"cols":[['
        '"Swap","X","Swap"]]}'
    ) == cirq.Circuit.from_ops(cirq.SWAP(a, c), cirq.X(b))
    with pytest.raises(ValueError, match='number of swap gates'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[['
            '"Swap"]]}'
        )
    with pytest.raises(ValueError, match='number of swap gates'):
        _ = quirk_url_to_circuit(
            'https://algassert.com/quirk#circuit={"cols":[['
            '"Swap","Swap","Swap"]]}'
        )


def assert_url_to_circuit_returns(
        json_text: str,
        expected_circuit: 'cirq.Circuit',
        *,
        output_amplitudes_from_quirk: List[Dict[str, float]] = None):
    circuit = quirk_url_to_circuit(
        f'https://algassert.com/quirk#circuit={json_text}')
    cirq.testing.assert_same_circuits(circuit, expected_circuit)
    if output_amplitudes_from_quirk is not None:
        expected = np.array([
            float(e['r']) + 1j * float(e['i'])
            for e in output_amplitudes_from_quirk
        ])

        # Swap endian-ness.
        n = len(circuit.all_qubits())
        expected = expected.reshape((2,) * n
                                    ).transpose(range(n)[::-1]).reshape(1 << n)

        np.testing.assert_allclose(cirq.final_wavefunction(circuit),
                                   expected,
                                   atol=1e-8)


def test_assert_url_to_circuit_returns():
    a, b = cirq.LineQubit.range(2)

    assert_url_to_circuit_returns(
        '{"cols":[["X","X"],["X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a),
            cirq.X(b),
            cirq.X(a),
        ),
        output_amplitudes_from_quirk=[
            {"r": 0, "i": 0},
            {"r": 0, "i": 0},
            {"r": 1, "i": 0},
            {"r": 0, "i": 0}
        ])

    assert_url_to_circuit_returns(
        '{"cols":[["X","X"],["X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a),
            cirq.X(b),
            cirq.X(a),
        ))

    with pytest.raises(AssertionError, match='Not equal to tolerance'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","X"],["X"]]}',
            cirq.Circuit.from_ops(
                cirq.X(a),
                cirq.X(b),
                cirq.X(a),
            ),
            output_amplitudes_from_quirk=[
                {"r": 0, "i": 0},
                {"r": 0, "i": -1},
                {"r": 0, "i": 0},
                {"r": 0, "i": 0}
            ])

    with pytest.raises(AssertionError, match='differs from expected circuit'):
        assert_url_to_circuit_returns(
            '{"cols":[["X","X"],["X"]]}',
            cirq.Circuit.from_ops(
                cirq.X(a),
                cirq.Y(b),
                cirq.X(a),
            ))


def test_gate_type_controls():
    a, b, c = cirq.LineQubit.range(3)

    assert_url_to_circuit_returns(
        '{"cols":[["•","X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(b).controlled_by(a),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["◦","X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a),
            cirq.X(b).controlled_by(a),
            cirq.X(a),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["⊕","X"]]}',
        cirq.Circuit.from_ops(
            cirq.Y(a)**0.5,
            cirq.X(b).controlled_by(a),
            cirq.Y(a)**-0.5,
        ),
        output_amplitudes_from_quirk=[
            {"r":0.5,"i":0},
            {"r":-0.5,"i":0},
            {"r":0.5,"i":0},
            {"r":0.5,"i":0},
        ])
    assert_url_to_circuit_returns(
        '{"cols":[["⊖","X"]]}',
        cirq.Circuit.from_ops(
            cirq.Y(a)**-0.5,
            cirq.X(b).controlled_by(a),
            cirq.Y(a)**+0.5,
        ),
        output_amplitudes_from_quirk=[
            {"r":0.5,"i":0},
            {"r":0.5,"i":0},
            {"r":0.5,"i":0},
            {"r":-0.5,"i":0},
        ])

    assert_url_to_circuit_returns(
        '{"cols":[["⊗","X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a)**-0.5,
            cirq.X(b).controlled_by(a),
            cirq.X(a)**+0.5,
        ),
        output_amplitudes_from_quirk=[
            {"r":0.5,"i":0},
            {"r":0,"i":-0.5},
            {"r":0.5,"i":0},
            {"r":0,"i":0.5},
        ])
    assert_url_to_circuit_returns(
        '{"cols":[["(/)","X"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a)**+0.5,
            cirq.X(b).controlled_by(a),
            cirq.X(a)**-0.5,
        ),
        output_amplitudes_from_quirk=[
            {"r":0.5,"i":0},
            {"r":0,"i":0.5},
            {"r":0.5,"i":0},
            {"r":0,"i":-0.5},
        ])

    qs = cirq.LineQubit.range(8)
    assert_url_to_circuit_returns(
        '{"cols":[["X","•","◦","⊕","⊖","⊗","(/)","Z"]]}',
        cirq.Circuit.from_ops(
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
        cirq.Circuit.from_ops(
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

    assert_url_to_circuit_returns(
        '{"cols":[["…"]]}',
        cirq.Circuit.from_ops(cirq.I.on(a)))

    assert_url_to_circuit_returns(
        '{"cols":[["NeGate"]]}',
        cirq.Circuit.from_ops(cirq.GlobalPhaseOperation(-1)))

    assert_url_to_circuit_returns(
        '{"cols":[["i"]]}',
        cirq.Circuit.from_ops(cirq.GlobalPhaseOperation(1j)))

    assert_url_to_circuit_returns(
        '{"cols":[["-i"]]}',
        cirq.Circuit.from_ops(cirq.GlobalPhaseOperation(-1j)))

    assert_url_to_circuit_returns(
        '{"cols":[["√i"]]}',
        cirq.Circuit.from_ops(cirq.GlobalPhaseOperation(1j**0.5)))

    assert_url_to_circuit_returns(
        '{"cols":[["√-i"]]}',
        cirq.Circuit.from_ops(cirq.GlobalPhaseOperation(1j**-0.5)))


def test_fixed_single_qubit_rotations():
    a, b, c, d = cirq.LineQubit.range(4)
    t = sympy.Symbol('t')

    assert_url_to_circuit_returns(
        '{"cols":[["H","X","Y","Z"]]}',
        cirq.Circuit.from_ops(cirq.H(a), cirq.X(b), cirq.Y(c), cirq.Z(d)))

    assert_url_to_circuit_returns(
        '{"cols":[["X^½","X^⅓","X^¼"],'
        '["X^⅛","X^⅟₁₆","X^⅟₃₂"],'
        '["X^-½","X^-⅓","X^-¼"],'
        '["X^-⅛","X^-⅟₁₆","X^-⅟₃₂"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a)**(1/2), cirq.X(b)**(1/3), cirq.X(c)**(1/4),
            cirq.X(a)**(1/8), cirq.X(b)**(1/16), cirq.X(c)**(1/32),
            cirq.X(a) ** (-1 / 2), cirq.X(b) ** (-1 / 3), cirq.X(c) ** (-1 / 4),
            cirq.X(a) ** (-1 / 8), cirq.X(b) ** (-1 / 16), cirq.X(c) ** (-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Y^½","Y^⅓","Y^¼"],'
        '["Y^⅛","Y^⅟₁₆","Y^⅟₃₂"],'
        '["Y^-½","Y^-⅓","Y^-¼"],'
        '["Y^-⅛","Y^-⅟₁₆","Y^-⅟₃₂"]]}',
        cirq.Circuit.from_ops(
            cirq.Y(a)**(1/2), cirq.Y(b)**(1/3), cirq.Y(c)**(1/4),
            cirq.Y(a)**(1/8), cirq.Y(b)**(1/16), cirq.Y(c)**(1/32),
            cirq.Y(a) ** (-1 / 2), cirq.Y(b) ** (-1 / 3), cirq.Y(c) ** (-1 / 4),
            cirq.Y(a) ** (-1 / 8), cirq.Y(b) ** (-1 / 16), cirq.Y(c) ** (-1 / 32),
        ))

    assert_url_to_circuit_returns(
        '{"cols":[["Z^½","Z^⅓","Z^¼"],'
        '["Z^⅛","Z^⅟₁₆","Z^⅟₃₂"],'
        '["Z^⅟₆₄","Z^⅟₁₂₈"],'
        '["Z^-½","Z^-⅓","Z^-¼"],'
        '["Z^-⅛","Z^-⅟₁₆"]]}',
        cirq.Circuit.from_ops(
            cirq.Z(a)**(1/2), cirq.Z(b)**(1/3), cirq.Z(c)**(1/4),
            cirq.Z(a)**(1/8), cirq.Z(b)**(1/16), cirq.Z(c)**(1/32),
            cirq.Z(a)**(1/64), cirq.Z(b)**(1/128),
            cirq.Z(a) ** (-1 / 2), cirq.Z(b) ** (-1 / 3), cirq.Z(c) ** (-1 / 4),
            cirq.Z(a) ** (-1 / 8), cirq.Z(b) ** (-1 / 16),
        ))

    # Dynamic single qubit rotations.
    assert_url_to_circuit_returns(
        '{"cols":[["X^t","Y^t","Z^t"],["X^-t","Y^-t","Z^-t"]]}',
        cirq.Circuit.from_ops(
            cirq.X(a)**t, cirq.Y(b)**t, cirq.Z(c)**t,
            cirq.X(a)**-t, cirq.Y(b)**-t, cirq.Z(c)**-t,
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["e^iXt","e^iYt","e^iZt"],["e^-iXt","e^-iYt","e^-iZt"]]}',
        cirq.Circuit.from_ops(
            cirq.Rx(2 * sympy.pi * t).on(a),
            cirq.Ry(2 * sympy.pi * t).on(b),
            cirq.Rz(2 * sympy.pi * t).on(c),
            cirq.Rx(2 * sympy.pi * -t).on(a),
            cirq.Ry(2 * sympy.pi * -t).on(b),
            cirq.Rz(2 * sympy.pi * -t).on(c),
        ))

    # Classically parameterized single qubit rotations.
    assert_url_to_circuit_returns(
        '{"cols":[["X^ft",{"id":"X^ft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.X(a)**sympy.sin(sympy.pi * t),
            cirq.X(b)**(t*t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Y^ft",{"id":"Y^ft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.Y(a)**sympy.sin(sympy.pi * t),
            cirq.Y(b)**(t*t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Z^ft",{"id":"Z^ft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.Z(a)**sympy.sin(sympy.pi * t),
            cirq.Z(b)**(t*t),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rxft",{"id":"Rxft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.Rx(sympy.pi * t * t).on(a),
            cirq.Rx(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Ryft",{"id":"Ryft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.Ry(sympy.pi * t * t).on(a),
            cirq.Ry(t * t).on(b),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["Rzft",{"id":"Rzft","arg":"t*t"}]]}',
        cirq.Circuit.from_ops(
            cirq.Rz(sympy.pi * t * t).on(a),
            cirq.Rz(t * t).on(b),
        ))

    # Displays.
    assert_url_to_circuit_returns(
        '{"cols":[["Amps2"],[1,"Amps3"],["Chance"],'
        '["Chance2"],["Density"],["Density3"],'
        '["Sample4"],["Bloch"],["Sample2"]'
        ']}',
        cirq.Circuit())

    # # Arithmetic.
    # yield from reg_arithmetic_gate("^A<B", 1, lambda x, a, b: x ^ int(a < b))
    # yield from reg_arithmetic_gate("^A>B", 1, lambda x, a, b: x ^ int(a > b))
    # yield from reg_arithmetic_gate("^A<=B", 1, lambda x, a, b: x ^ int(a <= b))
    # yield from reg_arithmetic_gate("^A>=B", 1, lambda x, a, b: x ^ int(a >= b))
    # yield from reg_arithmetic_gate("^A=B", 1, lambda x, a, b: x ^ int(a == b))
    # yield from reg_arithmetic_gate("^A!=B", 1, lambda x, a, b: x ^ int(a != b))
    # yield from reg_arithmetic_family("inc", lambda x: x + 1)
    # yield from reg_arithmetic_family("dec", lambda x: x - 1)
    # yield from reg_arithmetic_family(
    #     "incmodR", lambda x, r: (x + 1) % r if x < r else x)
    # yield from reg_arithmetic_family(
    #     "decmodR", lambda x, r: (x - 1) % r if x < r else x)
    # yield from reg_arithmetic_family("+=A", lambda x, a: x + a)
    # yield from reg_arithmetic_family("-=A", lambda x, a: x - a)
    # yield from reg_arithmetic_family(
    #     "+AmodR", lambda x, a, r: (x + a) % r if x < r else x)
    # yield from reg_arithmetic_family(
    #     "-AmodR", lambda x, a, r: (x - a) % r if x < r else x)
    # yield from reg_arithmetic_family(
    #     "+ABmodR", lambda x, a, b, r: (x + a * b) % r if x < r else x)
    # yield from reg_arithmetic_family(
    #     "-ABmodR", lambda x, a, b, r: (x - a * b) % r if x < r else x)
    # yield from reg_arithmetic_family("+=AA", lambda x, a: x + a * a)
    # yield from reg_arithmetic_family("-=AA", lambda x, a: x - a * a)
    # yield from reg_arithmetic_family("+=AB", lambda x, a, b: x + a * b)
    # yield from reg_arithmetic_family("-=AB", lambda x, a, b: x - a * b)
    # yield from reg_arithmetic_family("^=A", lambda x, a: x ^ a)
    # yield from reg_arithmetic_family("+cntA", lambda x, a: x + popcnt(a))
    # yield from reg_arithmetic_family("-cntA", lambda x, a: x - popcnt(a))
    # yield from reg_arithmetic_family(
    #     "Flip<A", lambda x, a: a - x - 1 if x < a else x)
    # yield from reg_arithmetic_family(
    #     "*AmodR", lambda x, a, r: (x * a) % r
    #     if x < r and modular_multiplicative_inverse(a, r) else x)
    # yield from reg_arithmetic_family(
    #     "/AmodR", lambda x, a, r: (x * (modular_multiplicative_inverse(a, r) or
    #                                     1)) % r if x < r else x)
    # yield from reg_arithmetic_family(
    #     "*BToAmodR", lambda x, a, b, r: (x * pow(b, a, r)) % r
    #     if x < r and modular_multiplicative_inverse(b, r) else x)
    # yield from reg_arithmetic_family(
    #     "/BToAmodR", lambda x, a, b, r: (x * pow(
    #         modular_multiplicative_inverse(b, r) or 1, a, r)) % r
    #     if x < r else x)
    # yield from reg_arithmetic_family("*A", lambda x, a: x * a if a & 1 else x)
    # yield from reg_size_dependent_arithmetic_family(
    #     "/A", lambda n: lambda x, a: x * modular_multiplicative_inverse(
    #         a, 1 << n) if a & 1 else x)
    #
    # # Dynamic gates with discretized actions.
    # yield from reg_unsupported_gates("X^⌈t⌉",
    #                                  "X^⌈t-¼⌉",
    #                                  reason="discrete parameter")
    # yield from reg_unsupported_family("Counting", reason="discrete parameter")
    # yield from reg_unsupported_family("Uncounting", reason="discrete parameter")
    # yield from reg_unsupported_family(">>t", reason="discrete parameter")
    # yield from reg_unsupported_family("<<t", reason="discrete parameter")
    #
    # # Gates that are no longer in the toolbox and have dominant replacements.
    # yield from reg_unsupported_family("add",
    #                                   reason="deprecated; use +=A instead")
    # yield from reg_unsupported_family("sub",
    #                                   reason="deprecated; use -=A instead")
    # yield from reg_unsupported_family("c+=ab",
    #                                   reason="deprecated; use +=AB instead")
    # yield from reg_unsupported_family("c-=ab",
    #                                   reason="deprecated; use -=AB instead")
    #
    # # Frequency space.
    # yield from reg_family("QFT", lambda n: cirq.QuantumFourierTransformGate(n))
    # yield from reg_family(
    #     "QFT†", lambda n: cirq.inverse(cirq.QuantumFourierTransformGate(n)))
    # yield from reg_family(
    #     "PhaseGradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
    #                                                       exponent=0.5))
    # yield from reg_family(
    #     "PhaseUngradient", lambda n: cirq.PhaseGradientGate(num_qubits=n,
    #                                                         exponent=-0.5))
    # yield from reg_family(
    #     "grad^t", lambda n: cirq.PhaseGradientGate(
    #         num_qubits=n, exponent=2**(n - 1) * sympy.Symbol('t')))
    # yield from reg_family(
    #     "grad^-t", lambda n: cirq.PhaseGradientGate(
    #         num_qubits=n, exponent=-2**(n - 1) * sympy.Symbol('t')))
    #
    # # Bit level permutations.
    # yield from reg_bit_permutation_family("<<", lambda n, x: (x + 1) % n)
    # yield from reg_bit_permutation_family(">>", lambda n, x: (x - 1) % n)
    # yield from reg_bit_permutation_family("rev", lambda n, x: n - x - 1)
    # yield from reg_bit_permutation_family("weave", interleave_bit)
    # yield from reg_bit_permutation_family("split", deinterleave_bit)
    # # Input gates.
    # yield from reg_input_family("inputA", "a")
    # yield from reg_input_family("inputB", "b")
    # yield from reg_input_family("inputR", "r")
    # yield from reg_input_family("revinputA", "a", rev=True)
    # yield from reg_input_family("revinputB", "b", rev=True)
    #
    # yield from reg_unsupported_gates("setA",
    #                                  "setB",
    #                                  "setR",
    #                                  reason="Cross column effects.")
    #
    # # Post selection.
    # yield from reg_unsupported_gates(
    #     "|0⟩⟨0|",
    #     "|1⟩⟨1|",
    #     "|+⟩⟨+|",
    #     "|-⟩⟨-|",
    #     "|X⟩⟨X|",
    #     "|/⟩⟨/|",
    #     "0",
    #     reason='postselection is not implemented in Cirq')
    # # Measurement.
    # yield from reg_gate("Measure", gate=ops.MeasurementGate(num_qubits=1))
    # yield from reg_gate("ZDetector", gate=ops.MeasurementGate(num_qubits=1))
    # yield from reg_gate("YDetector",
    #                     gate=ops.MeasurementGate(num_qubits=1),
    #                     basis_change=ops.X**-0.5)
    # yield from reg_gate("XDetector",
    #                     gate=ops.MeasurementGate(num_qubits=1),
    #                     basis_change=ops.H)
    # yield from reg_unsupported_gates(
    #     "XDetectControlReset",
    #     "YDetectControlReset",
    #     "ZDetectControlReset",
    #     reason="Classical feedback is not implemented in Cirq")
    #
    # # Quantum parameterized single qubit rotations.
    # yield from reg_parameterized_gate("X^(A/2^n)", ops.X, +1)
    # yield from reg_parameterized_gate("Y^(A/2^n)", ops.Y, +1)
    # yield from reg_parameterized_gate("Z^(A/2^n)", ops.Z, +1)
    # yield from reg_parameterized_gate("X^(-A/2^n)", ops.X, -1)
    # yield from reg_parameterized_gate("Y^(-A/2^n)", ops.Y, -1)
    # yield from reg_parameterized_gate("Z^(-A/2^n)", ops.Z, -1)
    #
