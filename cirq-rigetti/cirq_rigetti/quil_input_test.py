# Copyright 2020 The Cirq Developers
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

from inspect import signature

import numpy as np
import pytest

from pyquil.quil import Program
from pyquil.quilbase import Parameter, DefGate
from pyquil.quilatom import quil_cos, quil_sin, quil_exp
from pyquil.simulation import matrices
from pyquil.simulation.tools import program_unitary

import sympy
import cirq
from cirq import Circuit, LineQubit
from cirq import Simulator, unitary
from cirq.linalg.predicates import allclose_up_to_global_phase
from cirq_rigetti.quil_input import (
    UndefinedQuilGate,
    UnsupportedQuilInstruction,
    SUPPORTED_GATES,
    PARAMETRIC_TRANSFORMERS,
    CPHASE00,
    CPHASE01,
    CPHASE10,
    PSWAP,
    circuit_from_quil,
    defgate_to_cirq,
)

from cirq.ops.common_gates import CNOT, CZ, CZPowGate, H, S, T, ZPowGate, YPowGate, XPowGate
from cirq.ops.pauli_gates import X, Y, Z
from cirq.ops.identity import I
from cirq.ops.measurement_gate import MeasurementGate
from cirq.ops.swap_gates import ISWAP, ISwapPowGate, SWAP
from cirq.ops.three_qubit_gates import CCNOT, CSWAP


def test_gate_conversion():
    """Check that the gates all convert with matching unitaries."""
    for quil_gate, cirq_gate in SUPPORTED_GATES.items():
        if quil_gate in PARAMETRIC_TRANSFORMERS:
            pyquil_def = getattr(matrices, quil_gate)
            sig = signature(pyquil_def)
            num_params = len(sig.parameters)
            sample_params = list(np.random.random(num_params) * np.pi)

            pyquil_unitary = pyquil_def(*sample_params)
            cirq_unitary = unitary(cirq_gate(**PARAMETRIC_TRANSFORMERS[quil_gate](*sample_params)))
            assert np.allclose(pyquil_unitary, cirq_unitary)

        else:
            assert np.allclose(getattr(matrices, quil_gate), unitary(cirq_gate))


QUIL_PROGRAM = """
DECLARE ro BIT[3]
I 0
I 1
I 2
X 0
Y 1
Z 2
H 0
S 1
T 2
PHASE(pi/8) 0
PHASE(pi/8) 1
PHASE(pi/8) 2
RX(pi/2) 0
RY(pi/2) 1
RZ(pi/2) 2
CZ 0 1
CNOT 1 2
CPHASE(pi/2) 0 1
CPHASE00(pi/2) 1 2
CPHASE01(pi/2) 0 1
CPHASE10(pi/2) 1 2
ISWAP 0 1
PSWAP(pi/2) 1 2
SWAP 0 1
XY(pi/2) 1 2
CCNOT 0 1 2
CSWAP 0 1 2
MEASURE 0 ro[0]
MEASURE 1 ro[1]
MEASURE 2 ro[2]
"""


def test_circuit_from_quil():
    """Convert a test circuit from Quil with a wide range of gates."""
    q0, q1, q2 = LineQubit.range(3)
    cirq_circuit = Circuit(
        [
            I(q0),
            I(q1),
            I(q2),
            X(q0),
            Y(q1),
            Z(q2),
            H(q0),
            S(q1),
            T(q2),
            ZPowGate(exponent=1 / 8)(q0),
            ZPowGate(exponent=1 / 8)(q1),
            ZPowGate(exponent=1 / 8)(q2),
            XPowGate(exponent=1 / 2, global_shift=-0.5)(q0),
            YPowGate(exponent=1 / 2, global_shift=-0.5)(q1),
            ZPowGate(exponent=1 / 2, global_shift=-0.5)(q2),
            CZ(q0, q1),
            CNOT(q1, q2),
            CZPowGate(exponent=1 / 2, global_shift=0.0)(q0, q1),
            CPHASE00(phi=np.pi / 2)(q1, q2),
            CPHASE01(phi=np.pi / 2)(q0, q1),
            CPHASE10(phi=np.pi / 2)(q1, q2),
            ISWAP(q0, q1),
            PSWAP(phi=np.pi / 2)(q1, q2),
            SWAP(q0, q1),
            ISwapPowGate(exponent=1 / 2, global_shift=0.0)(q1, q2),
            CCNOT(q0, q1, q2),
            CSWAP(q0, q1, q2),
            MeasurementGate(1, key="ro[0]")(q0),
            MeasurementGate(1, key="ro[1]")(q1),
            MeasurementGate(1, key="ro[2]")(q2),
        ]
    )
    # build the same Circuit, using Quil
    quil_circuit = circuit_from_quil(Program(QUIL_PROGRAM))
    # test Circuit equivalence
    assert cirq_circuit == quil_circuit

    pyquil_circuit = Program(QUIL_PROGRAM)
    # drop declare and measures, get Program unitary
    pyquil_unitary = program_unitary(pyquil_circuit[1:-3], n_qubits=3)
    # fix qubit order convention
    cirq_circuit_swapped = Circuit(SWAP(q0, q2), cirq_circuit[:-1], SWAP(q0, q2))
    # get Circuit unitary
    cirq_unitary = cirq_circuit_swapped.unitary()
    # test unitary equivalence
    assert np.isclose(pyquil_unitary, cirq_unitary).all()


QUIL_PROGRAM_WITH_DEFGATE = """
DEFGATE MYZ:
    1,0
    0,-1

X 0
MYZ 0
"""


def test_quil_with_defgate():
    """Convert a Quil program with a DefGate."""
    q0 = LineQubit(0)
    cirq_circuit = Circuit([X(q0), Z(q0)])
    quil_circuit = circuit_from_quil(Program(QUIL_PROGRAM_WITH_DEFGATE))
    assert np.isclose(quil_circuit.unitary(), cirq_circuit.unitary()).all()


QUIL_PROGRAM_WITH_PARAMETERIZED_DEFGATE = """
DEFGATE MYPHASE(%phi):
    1,0
    0,EXP(i*%phi)

X 0
MYPHASE(pi/2) 0
"""


def test_program_with_parameterized_defgate():
    """Convert a Quil program with a parameterized DefGate."""
    program = Program(QUIL_PROGRAM_WITH_PARAMETERIZED_DEFGATE)
    circuit = circuit_from_quil(program)

    pyquil_unitary = np.array([[1, 0], [0, np.exp(1j * np.pi / 2)]]) @ matrices.X
    cirq_unitary = circuit.unitary()

    assert allclose_up_to_global_phase(pyquil_unitary, cirq_unitary, atol=1e-8)


def test_unsupported_quil_instruction():
    """Convert a program with invalid or unsupported instructions."""
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("NOP")

    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("RESET")


def test_undefined_quil_gate():
    """There are no such things as FREDKIN & TOFFOLI in Quil. The standard
    names for those gates in Quil are CSWAP and CCNOT. Of course, they can
    be defined via DEFGATE / DEFCIRCUIT.
    """
    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil("FREDKIN 0 1 2")

    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil("TOFFOLI 0 1 2")


QUIL_PROGRAM_WITH_PARAMETERS = """
DECLARE theta REAL[4]
RX(pi) 0
RX(theta[0]) 1
RX(2*theta[1]) 3
RX(2*theta[2] + 1) 2
RX(2*COS(theta[3])*EXP(i*theta[3])) 4
"""


def test_parametric_quil():
    """Convert a program which uses parameters and expressions."""
    program = Program(QUIL_PROGRAM_WITH_PARAMETERS)

    circuit = circuit_from_quil(program)

    q0, q1, q2, q3, q4 = LineQubit.range(5)
    theta_0, theta_1, theta_2, theta_3 = (
        sympy.Symbol("theta_0"),
        sympy.Symbol("theta_1"),
        sympy.Symbol("theta_2"),
        sympy.Symbol("theta_3"),
    )
    cirq_circuit = Circuit(
        [
            XPowGate(exponent=1, global_shift=-0.5)(q0),
            XPowGate(exponent=theta_0 / np.pi, global_shift=-0.5)(q1),
            XPowGate(exponent=(2 / np.pi) * theta_1, global_shift=-0.5)(q3),
            XPowGate(exponent=(2 / np.pi) * theta_2 + 1 / np.pi, global_shift=-0.5)(q2),
            XPowGate(
                exponent=(2 / np.pi) * sympy.cos(theta_3) * sympy.exp(1j * theta_3),
                global_shift=-0.5,
            )(q4),
        ]
    )

    assert cirq_circuit == circuit


QUIL_PROGRAM_WITH_SINGLE_PARAMETER = """
DECLARE theta REAL[1]
RZ(theta) 0
"""


def test_scalar_param_name_unchanged_by_conversion():
    """Ensure the name of scalar parameter does not change during conversion."""
    program = Program(QUIL_PROGRAM_WITH_SINGLE_PARAMETER)
    circuit = circuit_from_quil(program)
    assert cirq.parameter_names(circuit) == {"theta"}


def test_measurement_without_classical_reg():
    """Measure operations must declare a classical register."""
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("MEASURE 0")


# Insert a similar test for Kraus ops


QUIL_PROGRAM_WITH_READOUT_NOISE = """
DECLARE ro BIT[1]
RX(pi) 0
PRAGMA READOUT-POVM 0 "(0.9 0.050000000000000044 0.09999999999999998 0.95)"
MEASURE 0 ro[0]
"""


def test_readout_noise():
    """Convert a program with readout noise."""
    program = Program(QUIL_PROGRAM_WITH_READOUT_NOISE)
    circuit = circuit_from_quil(program)

    result = Simulator(seed=0).run(circuit, repetitions=1000)
    assert result.histogram(key="ro[0]")[1] < 1000
    assert result.histogram(key="ro[0]")[1] > 900


def test_resolve_parameters():
    """Test that parameters are correctly resolved for defined parametric gate."""
    theta, beta = Parameter("theta"), Parameter("beta")
    xy_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, quil_cos(theta / 2), 1j * quil_sin(theta / 2) * quil_exp(1j * beta), 0],
            [0, 1j * quil_sin(theta / 2) * quil_exp(1j * beta), 1j * quil_cos(theta / 2), 0],
            [0, 0, 0, 1],
        ]
    )

    defgate = DefGate("PHASEDXY", xy_matrix, parameters=[beta, theta])

    cirq_phased_xy = defgate_to_cirq(defgate)

    op = cirq_phased_xy(beta=sympy.Symbol("beta"), theta=sympy.Symbol("theta"))(
        cirq.LineQubit(0), cirq.LineQubit(1)
    )

    op._resolve_parameters_({"beta": 1.0, "theta": 2.0}, True)


def test_op_identifier():
    """Check that converted parametric defgates will be correctly identified."""
    theta, beta = Parameter("theta"), Parameter("beta")
    xy_matrix = np.array(
        [
            [1, 0, 0, 0],
            [0, quil_cos(theta / 2), 1j * quil_sin(theta / 2) * quil_exp(1j * beta), 0],
            [0, 1j * quil_sin(theta / 2) * quil_exp(1j * beta), 1j * quil_cos(theta / 2), 0],
            [0, 0, 0, 1],
        ]
    )

    defgate = DefGate("PHASEDXY", xy_matrix, parameters=[beta, theta])

    gate1 = defgate_to_cirq(defgate)
    gate2 = defgate_to_cirq(defgate)

    op = gate1(beta=np.pi, theta=np.pi)(cirq.LineQubit(0), cirq.LineQubit(1))

    assert op in cirq.OpIdentifier(gate1)
    assert op in cirq.OpIdentifier(gate2)
