import pytest

import cirq
from cirq.contrib.svg import circuit_to_svg


def test_svg():
    a, b, c = cirq.LineQubit.range(3)

    svg_text = circuit_to_svg(
        cirq.Circuit(
            cirq.CNOT(a, b), cirq.CZ(b, c), cirq.SWAP(a, c),
            cirq.PhasedXPowGate(exponent=0.123, phase_exponent=0.456).on(c),
            cirq.Z(a), cirq.measure(a, b, c, key='z')))
    assert '<svg' in svg_text
    assert '</svg>' in svg_text


def test_validation():
    with pytest.raises(ValueError):
        circuit_to_svg(cirq.Circuit())

    q0 = cirq.LineQubit(0)
    with pytest.raises(ValueError):
        circuit_to_svg(
            cirq.Circuit([cirq.Moment([cirq.X(q0)]),
                          cirq.Moment([])]))
