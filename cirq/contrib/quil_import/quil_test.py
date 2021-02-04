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

import numpy as np
import pytest

from pyquil import Program
from pyquil.simulation.tools import program_unitary

from cirq import Circuit, LineQubit
from cirq.contrib.quil_import.quil import (
    UndefinedQuilGate,
    UnsupportedQuilInstruction,
    circuit_from_quil,
    cphase,
    cphase00,
    cphase01,
    cphase10,
    pswap,
    xy,
)
from cirq.ops import (
    CCNOT,
    CNOT,
    CSWAP,
    CZ,
    H,
    I,
    ISWAP,
    MeasurementGate,
    S,
    SWAP,
    T,
    X,
    Y,
    Z,
    rx,
    ry,
    rz,
)

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
            Z(q0) ** (1 / 8),
            Z(q1) ** (1 / 8),
            Z(q2) ** (1 / 8),
            rx(np.pi / 2)(q0),
            ry(np.pi / 2)(q1),
            rz(np.pi / 2)(q2),
            CZ(q0, q1),
            CNOT(q1, q2),
            cphase(np.pi / 2)(q0, q1),
            cphase00(np.pi / 2)(q1, q2),
            cphase01(np.pi / 2)(q0, q1),
            cphase10(np.pi / 2)(q1, q2),
            ISWAP(q0, q1),
            pswap(np.pi / 2)(q1, q2),
            SWAP(q0, q1),
            xy(np.pi / 2)(q1, q2),
            CCNOT(q0, q1, q2),
            CSWAP(q0, q1, q2),
            MeasurementGate(1, key="ro[0]")(q0),
            MeasurementGate(1, key="ro[1]")(q1),
            MeasurementGate(1, key="ro[2]")(q2),
        ]
    )
    # build the same Circuit, using Quil
    quil_circuit = circuit_from_quil(QUIL_PROGRAM)
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
    q0 = LineQubit(0)
    cirq_circuit = Circuit([X(q0), Z(q0)])
    quil_circuit = circuit_from_quil(QUIL_PROGRAM_WITH_DEFGATE)
    assert np.isclose(quil_circuit.unitary(), cirq_circuit.unitary()).all()


QUIL_PROGRAM_WITH_PARAMETERIZED_DEFGATE = """
DEFGATE MYPHASE(%phi):
    1,0
    0,EXP(i*%phi)

X 0
MYPHASE 0
"""


def test_unsupported_quil_instruction():
    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("NOP")

    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("PRAGMA ADD-KRAUS X 0 \"(0.0 1.0 1.0 0.0)\"")

    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil("RESET")

    with pytest.raises(UnsupportedQuilInstruction):
        circuit_from_quil(QUIL_PROGRAM_WITH_PARAMETERIZED_DEFGATE)


def test_undefined_quil_gate():
    """There are no such things as FREDKIN & TOFFOLI in Quil. The standard
    names for those gates in Quil are CSWAP and CCNOT. Of course, they can
    be defined via DEFGATE / DEFCIRCUIT."""
    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil("FREDKIN 0 1 2")

    with pytest.raises(UndefinedQuilGate):
        circuit_from_quil("TOFFOLI 0 1 2")
