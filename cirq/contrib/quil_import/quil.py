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

from typing import Callable, Dict, Union

import numpy as np
from pyquil.parser import parse
from pyquil.quilbase import (
    Declare,
    DefGate,
    Gate as PyQuilGate,
    Measurement as PyQuilMeasurement,
    Pragma,
    Reset,
    ResetQubit,
)

from cirq import Circuit, LineQubit
from cirq.ops import (
    CCNOT,
    CNOT,
    CSWAP,
    CZ,
    CZPowGate,
    Gate,
    H,
    I,
    ISWAP,
    ISwapPowGate,
    MatrixGate,
    MeasurementGate,
    S,
    SWAP,
    T,
    TwoQubitDiagonalGate,
    X,
    Y,
    Z,
    ZPowGate,
    rx,
    ry,
    rz,
)


class UndefinedQuilGate(Exception):
    pass


class UnsupportedQuilInstruction(Exception):
    pass


#
# Functions for converting supported parameterized Quil gates.
#


def cphase(param: float) -> CZPowGate:
    """Returns a controlled-phase gate as a Cirq CZPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's CPHASE
    gate and the exponent of Cirq's CZPowGate differ by a factor of pi.

    Args:
        param: Gate parameter (in radians).

    Returns:
        A CZPowGate equivalent to a CPHASE gate of given angle.
    """
    return CZPowGate(exponent=param / np.pi)


def cphase00(phi: float) -> TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE00 gate.

    In pyQuil, CPHASE00(phi) = diag([exp(1j * phi), 1, 1, 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [phi, 0, 0, 0].

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE00 gate of given angle.
    """
    return TwoQubitDiagonalGate([phi, 0, 0, 0])


def cphase01(phi: float) -> TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE01 gate.

    In pyQuil, CPHASE01(phi) = diag(1, [exp(1j * phi), 1, 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [0, phi, 0, 0].

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE01 gate of given angle.
    """
    return TwoQubitDiagonalGate([0, phi, 0, 0])


def cphase10(phi: float) -> TwoQubitDiagonalGate:
    """Returns a Cirq TwoQubitDiagonalGate for pyQuil's CPHASE10 gate.

    In pyQuil, CPHASE10(phi) = diag(1, 1, [exp(1j * phi), 1]), and in Cirq,
    a TwoQubitDiagonalGate is specified by its diagonal in radians, which
    would be [0, 0, phi, 0].

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A TwoQubitDiagonalGate equivalent to a CPHASE10 gate of given angle.
    """
    return TwoQubitDiagonalGate([0, 0, phi, 0])


def phase(param: float) -> ZPowGate:
    """Returns a single-qubit phase gate as a Cirq ZPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's PHASE
    gate and the exponent of Cirq's ZPowGate differ by a factor of pi.

    Args:
        param: Gate parameter (in radians).

    Returns:
        A ZPowGate equivalent to a PHASE gate of given angle.
    """
    return ZPowGate(exponent=param / np.pi)


def pswap(phi: float) -> MatrixGate:
    """Returns a Cirq MatrixGate for pyQuil's PSWAP gate.

    Args:
        phi: Gate parameter (in radians).

    Returns:
        A MatrixGate equivalent to a PSWAP gate of given angle.
    """
    pswap_matrix = np.array([
        [1, 0, 0, 0],
        [0, 0, np.exp(1j * phi), 0],
        [0, np.exp(1j * phi), 0, 0],
        [0, 0, 0, 1],
    ],
                            dtype=complex)
    return MatrixGate(pswap_matrix)


def xy(param: float) -> ISwapPowGate:
    """Returns an ISWAP-family gate as a Cirq ISwapPowGate with exponent
    determined by the input param. The angle parameter of pyQuil's XY gate
    and the exponent of Cirq's ISwapPowGate differ by a factor of pi.

    Args:
        param: Gate parameter (in radians).

    Returns:
        An ISwapPowGate equivalent to an XY gate of given angle.
    """
    return ISwapPowGate(exponent=param / np.pi)


PRAGMA_ERROR = """
Please remove PRAGMAs from your Quil program.
If you would like to add noise, do so after conversion.
"""

RESET_ERROR = """
Please remove RESETs from your Quil program.
RESET directives have special meaning on QCS, to enable active reset.
"""

# Parameterized gates map to functions that produce Gate constructors.
SUPPORTED_GATES: Dict[str, Union[Gate, Callable[..., Gate]]] = {
    "CCNOT": CCNOT,
    "CNOT": CNOT,
    "CSWAP": CSWAP,
    "CPHASE": cphase,
    "CPHASE00": cphase00,
    "CPHASE01": cphase01,
    "CPHASE10": cphase10,
    "CZ": CZ,
    "PHASE": phase,
    "H": H,
    "I": I,
    "ISWAP": ISWAP,
    "PSWAP": pswap,
    "RX": rx,
    "RY": ry,
    "RZ": rz,
    "S": S,
    "SWAP": SWAP,
    "T": T,
    "X": X,
    "Y": Y,
    "Z": Z,
    "XY": xy,
}


def circuit_from_quil(quil: str) -> Circuit:
    """Convert a Quil program to a Cirq Circuit.

    Args:
        quil: The Quil program to convert.

    Returns:
        A Cirq Circuit generated from the Quil program.

    References:
        https://github.com/rigetti/pyquil
    """
    circuit = Circuit()
    defined_gates = SUPPORTED_GATES.copy()
    instructions = parse(quil)

    for inst in instructions:
        # Add DEFGATE-defined gates to defgates dict using MatrixGate.
        if isinstance(inst, DefGate):
            if inst.parameters:
                raise UnsupportedQuilInstruction(
                    "Parameterized DEFGATEs are currently unsupported.")
            defined_gates[inst.name] = MatrixGate(inst.matrix)

        # Pass when encountering a DECLARE.
        elif isinstance(inst, Declare):
            pass

        # Convert pyQuil gates to Cirq operations.
        elif isinstance(inst, PyQuilGate):
            quil_gate_name = inst.name
            quil_gate_params = inst.params
            line_qubits = list(LineQubit(q.index) for q in inst.qubits)
            if quil_gate_name not in defined_gates:
                raise UndefinedQuilGate(
                    f"Quil gate {quil_gate_name} not supported in Cirq.")
            cirq_gate_fn = defined_gates[quil_gate_name]
            if quil_gate_params:
                circuit += cirq_gate_fn(*quil_gate_params)(*line_qubits)
            else:
                circuit += cirq_gate_fn(*line_qubits)

        # Convert pyQuil MEASURE operations to Cirq MeasurementGate objects.
        elif isinstance(inst, PyQuilMeasurement):
            line_qubit = LineQubit(inst.qubit.index)
            quil_memory_reference = inst.classical_reg.out()
            circuit += MeasurementGate(1, key=quil_memory_reference)(line_qubit)

        # Raise a targeted error when encountering a PRAGMA.
        elif isinstance(inst, Pragma):
            raise UnsupportedQuilInstruction(PRAGMA_ERROR)

        # Raise a targeted error when encountering a RESET.
        elif isinstance(inst, (Reset, ResetQubit)):
            raise UnsupportedQuilInstruction(RESET_ERROR)

        # Raise a general error when encountering an unconsidered type.
        else:
            raise UnsupportedQuilInstruction(
                f"Quil instruction {inst} of type {type(inst)}"
                " not currently supported in Cirq.")

    return circuit
