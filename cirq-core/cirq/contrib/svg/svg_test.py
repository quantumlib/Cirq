# pylint: disable=wrong-or-nonexistent-copyright-notice
import IPython.display
import numpy as np
import pytest

import cirq
from cirq.contrib.svg import circuit_to_svg


def test_svg():
    a, b, c = cirq.LineQubit.range(3)

    svg_text = circuit_to_svg(
        cirq.Circuit(
            cirq.CNOT(a, b),
            cirq.CZ(b, c),
            cirq.SWAP(a, c),
            cirq.PhasedXPowGate(exponent=0.123, phase_exponent=0.456).on(c),
            cirq.Z(a),
            cirq.measure(a, b, c, key='z'),
            cirq.MatrixGate(np.eye(2)).on(a),
        )
    )
    assert '?' in svg_text
    assert '<svg' in svg_text
    assert '</svg>' in svg_text


def test_svg_noise():
    noise_model = cirq.ConstantQubitNoiseModel(cirq.DepolarizingChannel(p=1e-3))
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.X(q))
    circuit = cirq.Circuit(noise_model.noisy_moments(circuit.moments, [q]))
    svg = circuit_to_svg(circuit)
    assert '>D(0.001)</text>' in svg


def test_validation():
    with pytest.raises(ValueError):
        circuit_to_svg(cirq.Circuit())


def test_empty_moments():
    a, b = cirq.LineQubit.range(2)
    svg_1 = circuit_to_svg(
        cirq.Circuit(
            cirq.Moment(),
            cirq.Moment(cirq.CNOT(a, b)),
            cirq.Moment(),
            cirq.Moment(cirq.SWAP(a, b)),
            cirq.Moment(cirq.Z(a)),
            cirq.Moment(cirq.measure(a, b, key='z')),
            cirq.Moment(),
        )
    )
    assert '<svg' in svg_1
    assert '</svg>' in svg_1

    svg_2 = circuit_to_svg(cirq.Circuit(cirq.Moment()))
    assert '<svg' in svg_2
    assert '</svg>' in svg_2


@pytest.mark.parametrize(
    'symbol,svg_symbol',
    [
        ('<a', '&lt;a'),
        ('<=b', '&lt;=b'),
        ('>c', '&gt;c'),
        ('>=d', '&gt;=d'),
        ('>e<', '&gt;e&lt;'),
        ('A[<virtual>]B[cirq.VirtualTag()]C>D<E', 'ABC&gt;D&lt;E'),
    ],
)
def test_gate_with_less_greater_str(symbol, svg_symbol):
    class CustomGate(cirq.Gate):
        def _num_qubits_(self) -> int:
            return 1

        def _circuit_diagram_info_(self, _) -> cirq.CircuitDiagramInfo:
            return cirq.CircuitDiagramInfo(wire_symbols=[symbol])

    circuit = cirq.Circuit(CustomGate().on(cirq.LineQubit(0)))
    svg = circuit_to_svg(circuit)

    _ = IPython.display.SVG(svg)
    assert svg_symbol in svg
