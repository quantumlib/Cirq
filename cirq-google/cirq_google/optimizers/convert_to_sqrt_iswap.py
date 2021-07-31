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
from typing import List, Optional

import numpy as np
import sympy

import cirq

SQRT_ISWAP = cirq.ISWAP ** 0.5
SQRT_ISWAP_INV = cirq.ISWAP ** -0.5


# TODO: Combine this with the equivalent functions in google/gate_set.py
# Or better yet, write a proper gate set so we don't need this in two places.
# Github issue: https://github.com/quantumlib/Cirq/issues/2970
def _near_mod_n(e, t, n, atol=1e-8):
    return abs((e - t + 1) % n - 1) <= atol


def _near_mod_2pi(e, t, atol=1e-8):
    return _near_mod_n(e, t, 2 * np.pi, atol=atol)


class ConvertToSqrtIswapGates(cirq.PointOptimizer):
    """Attempts to convert gates into ISWAP**-0.5 gates.

    Since we have Z rotations and arbitrary XY rotations, we
    can rely on cirq decomposition for one qubit gates and
    need to only specify special decompositions for two qubit gates.

    Currently natively specified gates are CZPowGate, ISwapPowGate,
    and FSimGate.  This will also support gates that decompose into
    the above gates.
    """

    def __init__(self, ignore_failures=False) -> None:
        """Inits ConvertToSqrtIswapGates.

        Args:
            ignore_failures: If set, gates that fail to convert are forwarded
                unchanged. If not set, conversion failures raise a TypeError.
        """
        super().__init__()
        self.ignore_failures = ignore_failures

    # TODO(#3388) Add summary line to docstring.
    # pylint: disable=docstring-first-line-empty
    def _convert_one(self, op: cirq.Operation) -> cirq.OP_TREE:
        """
        Decomposer intercept:  Let cirq decompose one-qubit gates,
        intercept on 2-qubit gates if they are known gates.
        """
        if isinstance(op, cirq.GlobalPhaseOperation):
            return []

        gate = op.gate

        if len(op.qubits) != 2:
            return NotImplemented

        q0, q1 = op.qubits

        if isinstance(gate, cirq.CZPowGate):
            if isinstance(gate.exponent, sympy.Basic):
                return cphase_symbols_to_sqrt_iswap(q0, q1, gate.exponent)
            else:
                return cphase_to_sqrt_iswap(q0, q1, gate.exponent)
        if isinstance(gate, cirq.SwapPowGate):
            return swap_to_sqrt_iswap(q0, q1, gate.exponent)
        if isinstance(gate, cirq.ISwapPowGate):
            return iswap_to_sqrt_iswap(q0, q1, gate.exponent)
        if isinstance(gate, cirq.FSimGate):
            return fsim_gate(q0, q1, gate.theta, gate.phi)

        return NotImplemented

    # pylint: enable=docstring-first-line-empty
    def _on_stuck_raise(self, bad):
        return TypeError(
            f"Don't know how to work with {bad}. "
            "It isn't a native sqrt ISWAP operation, "
            "a 1 or 2 qubit gate with a known unitary, "
            "or composite."
        )

    def convert(self, op: cirq.Operation) -> List[cirq.Operation]:
        return cirq.decompose(
            op,
            keep=is_sqrt_iswap_compatible,
            intercepting_decomposer=self._convert_one,
            on_stuck_raise=(None if self.ignore_failures else self._on_stuck_raise),
        )

    def optimization_at(
        self, circuit: cirq.Circuit, index: int, op: cirq.Operation
    ) -> Optional[cirq.PointOptimizationSummary]:
        if isinstance(op.gate, cirq.MatrixGate) and len(op.qubits) == 1:
            return None

        converted = self.convert(op)
        if len(converted) == 1 and converted[0] is op:
            return None

        return cirq.PointOptimizationSummary(
            clear_span=1, new_operations=converted, clear_qubits=op.qubits
        )


def is_sqrt_iswap_compatible(op: cirq.Operation) -> bool:
    """Check if the given operation is compatible with the sqrt_iswap gateset
    gate set.

    Args:
        op: Input operation.

    Returns:
        True if the operation is native to the gate set, false otherwise.
    """
    return is_basic_gate(op.gate) or is_sqrt_iswap(op.gate)


def is_sqrt_iswap(gate: Optional[cirq.Gate]) -> bool:
    """Checks if this is a ± sqrt(iSWAP) gate specified using either
    ISwapPowGate or with the equivalent FSimGate.
    """
    if (
        isinstance(gate, cirq.FSimGate)
        and not isinstance(gate.theta, sympy.Basic)
        and _near_mod_2pi(abs(gate.theta), np.pi / 4)
        and _near_mod_2pi(gate.phi, 0)
    ):
        return True
    return (
        isinstance(gate, cirq.ISwapPowGate)
        and not isinstance(gate.exponent, sympy.Basic)
        and _near_mod_n(abs(gate.exponent), 0.5, 4)
    )


def is_basic_gate(gate: Optional[cirq.Gate]) -> bool:
    """Check if a gate is a basic supported one-qubit gate.

    Args:
        gate: Input gate.

    Returns:
        True if the gate is native to the gate set, false otherwise.
    """
    return isinstance(
        gate,
        (
            cirq.MeasurementGate,
            cirq.PhasedXZGate,
            cirq.PhasedXPowGate,
            cirq.XPowGate,
            cirq.YPowGate,
            cirq.ZPowGate,
        ),
    )


def cphase_to_sqrt_iswap(a, b, turns):
    """Implement a C-Phase gate using two sqrt ISWAP gates and single-qubit
    operations. The circuit is equivalent to cirq.CZPowGate(exponent=turns).

    Output unitary:
    [1   0   0   0],
    [0   1   0   0],
    [0   0   1   0],
    [0   0   0   e^{i turns pi}].

    Args:
        a: the first qubit
        b: the second qubit
        turns: Exponent specifying the evolution time in number of rotations.
    """
    theta = (turns % 2) * np.pi
    if 0 <= theta <= np.pi:
        sign = 1.0
        theta_prime = theta
    elif np.pi < theta < 2 * np.pi:
        sign = -1.0
        theta_prime = 2 * np.pi - theta

    if np.isclose(theta, np.pi):
        # If we are close to pi, just set values manually to avoid possible
        # numerical errors with arcsin of greater than 1.0 (Ahem, Windows).
        phi = np.pi / 2
        xi = np.pi / 2
    else:
        phi = np.arcsin(np.sqrt(2) * np.sin(theta_prime / 4))
        xi = np.arctan(np.tan(phi) / np.sqrt(2))

    yield cirq.rz(sign * 0.5 * theta_prime).on(a)
    yield cirq.rz(sign * 0.5 * theta_prime).on(b)
    yield cirq.rx(xi).on(a)
    yield cirq.X(b) ** (-sign * 0.5)
    yield SQRT_ISWAP_INV(a, b)
    yield cirq.rx(-2 * phi).on(a)
    yield SQRT_ISWAP(a, b)

    yield cirq.rx(xi).on(a)
    yield cirq.X(b) ** (sign * 0.5)
    # Corrects global phase
    yield cirq.GlobalPhaseOperation(np.exp(sign * theta_prime * 0.25j))


def cphase_symbols_to_sqrt_iswap(a, b, turns):
    """Version of cphase_to_sqrt_iswap that works with symbols.

    Note that the formulae contained below will need to be flattened
    into a sweep before serializing.
    """
    theta = sympy.Mod(turns, 2.0) * sympy.pi

    # -1 if theta > pi.  Adds a hacky fudge factor so theta=pi is not 0
    sign = sympy.sign(sympy.pi - theta + 1e-9)

    # For sign = 1: theta. For sign = -1, 2pi-theta
    theta_prime = (sympy.pi - sign * sympy.pi) + sign * theta

    phi = sympy.asin(np.sqrt(2) * sympy.sin(theta_prime / 4))
    xi = sympy.atan(sympy.tan(phi) / np.sqrt(2))

    yield cirq.rz(sign * 0.5 * theta_prime).on(a)
    yield cirq.rz(sign * 0.5 * theta_prime).on(b)
    yield cirq.rx(xi).on(a)
    yield cirq.X(b) ** (-sign * 0.5)
    yield SQRT_ISWAP_INV(a, b)
    yield cirq.rx(-2 * phi).on(a)
    yield SQRT_ISWAP(a, b)
    yield cirq.rx(xi).on(a)
    yield cirq.X(b) ** (sign * 0.5)


def iswap_to_sqrt_iswap(a, b, turns):
    """Implement the evolution of the hopping term using two sqrt_iswap gates
     and single-qubit operations. Output unitary:
    [1   0   0   0],
    [0   c  is   0],
    [0  is   c   0],
    [0   0   0   1],
    where c = cos(t * np.pi / 2) and s = sin(t * np.pi / 2).

    Args:
        a: the first qubit
        b: the second qubit
        t: Exponent that specifies the evolution time in number of rotations.
    """
    yield cirq.Z(a) ** 0.75
    yield cirq.Z(b) ** 0.25
    yield SQRT_ISWAP_INV(a, b)
    yield cirq.Z(a) ** (-turns / 2 + 1)
    yield cirq.Z(b) ** (turns / 2)
    yield SQRT_ISWAP_INV(a, b)
    yield cirq.Z(a) ** 0.25
    yield cirq.Z(b) ** -0.25


def swap_to_sqrt_iswap(a, b, turns):
    """Implement the evolution of the hopping term using two sqrt_iswap gates
     and single-qubit operations. Output unitary:
    [[1, 0,        0,     0],
     [0, g·c,    -i·g·s,  0],
     [0, -i·g·s,  g·c,    0],
     [0,   0,      0,     1]]
     where c = cos(theta) and s = sin(theta).
        Args:
            a: the first qubit
            b: the second qubit
            theta: The rotational angle that specifies the gate, where
            c = cos(π·t/2), s = sin(π·t/2), g = exp(i·π·t/2).
    """
    if not isinstance(turns, sympy.Basic) and _near_mod_n(turns, 1.0, 2):
        # Decomposition for cirq.SWAP
        yield cirq.Y(a) ** 0.5
        yield cirq.Y(b) ** 0.5
        yield SQRT_ISWAP(a, b)
        yield cirq.Y(a) ** -0.5
        yield cirq.Y(b) ** -0.5
        yield SQRT_ISWAP(a, b)
        yield cirq.X(a) ** -0.5
        yield cirq.X(b) ** -0.5
        yield SQRT_ISWAP(a, b)
        yield cirq.X(a) ** 0.5
        yield cirq.X(b) ** 0.5
        return

    yield cirq.Z(a) ** 1.25
    yield cirq.Z(b) ** -0.25
    yield cirq.ISWAP(a, b) ** -0.5
    yield cirq.Z(a) ** (-turns / 2 + 1)
    yield cirq.Z(b) ** (turns / 2)
    yield cirq.ISWAP(a, b) ** -0.5
    yield cirq.Z(a) ** (turns / 2 - 0.25)
    yield cirq.Z(b) ** (turns / 2 + 0.25)
    yield cirq.CZ.on(a, b) ** (-turns)


def fsim_gate(a, b, theta, phi):
    """FSimGate has a default decomposition in cirq to XXPowGate and YYPowGate,
    which is an awkward decomposition for this gate set.
    Decompose into ISWAP and CZ instead."""
    if theta != 0.0:
        yield cirq.ISWAP(a, b) ** (-2 * theta / np.pi)
    if phi != 0.0:
        yield cirq.CZPowGate(exponent=-phi / np.pi)(a, b)
