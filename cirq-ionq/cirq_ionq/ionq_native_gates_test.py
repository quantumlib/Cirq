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
"""Tests for IonQ native gates"""

import cirq
import numpy
import pytest

import cirq_ionq as ionq


PARAMS_FOR_ONE_ANGLE_GATE = [0, 0.1, 0.4, 0.5, 1, 2]
PARAMS_FOR_TWO_ANGLE_GATE = [(0, 1), (0.1, 1), (0.4, 1), (0.5, 0), (0, 1), (0.1, 2)]
INVALID_GATE_POWER = [-2, -0.5, 0, 0.5, 2]


@pytest.mark.parametrize(
    "gate,nqubits,diagram",
    [
        (ionq.GPIGate(phi=0.1), 1, "0: ───GPI(0.1)───"),
        (ionq.GPI2Gate(phi=0.2), 1, "0: ───GPI2(0.2)───"),
        (ionq.MSGate(phi0=0.1, phi1=0.2), 2, "0: ───MS(0.1)───\n      │\n1: ───MS(0.2)───"),
        (ionq.ZZGate(theta=0.3), 2, "0: ───ZZ(0.3)───\n      │\n1: ───ZZ────────"),
    ],
)
def test_gate_methods(gate, nqubits, diagram):
    assert str(gate) != ""
    assert repr(gate) != ""
    assert gate.num_qubits() == nqubits
    assert cirq.protocols.circuit_diagram_info(gate) is not None
    c = cirq.Circuit()
    c.append([gate.on(*cirq.LineQubit.range(nqubits))])
    assert c.to_text_diagram() == diagram


@pytest.mark.parametrize(
    "gate",
    [
        ionq.GPIGate(phi=0.1),
        ionq.GPI2Gate(phi=0.2),
        ionq.MSGate(phi0=0.1, phi1=0.2),
        ionq.ZZGate(theta=0.4),
    ],
)
def test_gate_json(gate):
    g_json = cirq.to_json(gate)
    assert cirq.read_json(json_text=g_json) == gate


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, 0.5, 1, 2])
def test_gpi_unitary(phase):
    """Tests that the GPI gate is unitary."""
    gate = ionq.GPIGate(phi=phase)

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, 0.5, 1, 2])
def test_gpi2_unitary(phase):
    """Tests that the GPI2 gate is unitary."""
    gate = ionq.GPI2Gate(phi=phase)

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(2))


@pytest.mark.parametrize("phases", [(0, 1), (0.1, 1), (0.4, 1), (0.5, 0), (0, 1), (0.1, 2)])
def test_ms_unitary(phases):
    """Tests that the MS gate is unitary."""
    gate = ionq.MSGate(phi0=phases[0], phi1=phases[1])

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(4))


@pytest.mark.parametrize("phase", [0, 0.1, 0.4, 0.5, 1, 2])
def test_zz_unitary(phase):
    """Tests that the ZZ gate is unitary."""
    gate = ionq.ZZGate(theta=phase)

    mat = cirq.protocols.unitary(gate)
    numpy.testing.assert_array_almost_equal(mat.dot(mat.conj().T), numpy.identity(4))


@pytest.mark.parametrize(
    "gate",
    [
        *[ionq.GPIGate(phi=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
        *[ionq.GPI2Gate(phi=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
        *[ionq.MSGate(phi0=angles[0], phi1=angles[1]) for angles in PARAMS_FOR_TWO_ANGLE_GATE],
        *[ionq.ZZGate(theta=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
    ],
)
def test_gate_inverse(gate):
    """Tests that the inverse of natives gate are correct."""
    mat = cirq.protocols.unitary(gate)
    mat_inverse = cirq.protocols.unitary(gate**-1)
    dim = mat.shape[0]

    numpy.testing.assert_array_almost_equal(mat.dot(mat_inverse), numpy.identity(dim))


@pytest.mark.parametrize(
    "gate",
    [
        *[ionq.GPIGate(phi=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
        *[ionq.GPI2Gate(phi=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
        *[ionq.ZZGate(theta=angle) for angle in PARAMS_FOR_ONE_ANGLE_GATE],
        *[ionq.MSGate(phi0=angles[0], phi1=angles[1]) for angles in PARAMS_FOR_TWO_ANGLE_GATE],
    ],
)
def test_gate_power1(gate):
    """Tests that power=1 for native gates are correct."""
    mat = cirq.protocols.unitary(gate)
    mat_power1 = cirq.protocols.unitary(gate**1)

    numpy.testing.assert_array_almost_equal(mat, mat_power1)


@pytest.mark.parametrize(
    "gate,power",
    [
        *[(ionq.GPIGate(phi=0.1), power) for power in INVALID_GATE_POWER],
        *[(ionq.GPI2Gate(phi=0.1), power) for power in INVALID_GATE_POWER],
        *[(ionq.MSGate(phi0=0.1, phi1=0.2), power) for power in INVALID_GATE_POWER],
        *[(ionq.ZZGate(theta=0.1), power) for power in INVALID_GATE_POWER],
    ],
)
def test_gate_power_not_implemented(gate, power):
    """Tests that any power other than 1 and -1 is not implemented."""
    with pytest.raises(TypeError):
        _ = gate**power
