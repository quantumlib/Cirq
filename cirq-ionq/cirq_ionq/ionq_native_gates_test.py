import cirq
import math
import numpy
import pytest
from .ionq_native_gates import GPIGate, GPI2Gate, MSGate

@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi / 2, math.pi, 2 * math.pi])
def test_gpi_unitary(phase):
    """Tests that the GPI gate is unitary."""
    gate = GPIGate(phi=phase)

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, math.pi / 2, math.pi, 2 * math.pi])
def test_gpi2_unitary(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = GPI2Gate(phi=phase)

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize(
    "phases",
    [(0, 1), (0.1, 1), (0.4, 1), (math.pi / 2, 0), (0, math.pi), (0.1, 2 * math.pi)],
)
def test_ms_unitary(phases):
    """Tests that the MS gate is unitary."""
    gate = MSGate(phi1=phases[0], phi2=phases[1])

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(4))
