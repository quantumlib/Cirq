# pylint: disable=wrong-or-nonexistent-copyright-notice
import pytest
import numpy as np

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
