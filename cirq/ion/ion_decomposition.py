# Copyright 2018 The Cirq Developers
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

"""
Utility methods related to optimizing quantum circuits using native iontrap operations.

Gate compilation methods implemented here are following the paper below:
    'Basic circuit compilation techniques for an ion-trap quantum machine'
    arXiv:1603.07678
"""

from typing import List, Tuple, Optional, cast, Union

import math
import cmath
import numpy as np

from cirq import ops, linalg, protocols, ion, decompositions, value


def two_qubit_matrix_to_operations(q0: ops.QubitId,
                                   q1: ops.QubitId,
                                   mat: np.ndarray,
                                   tolerance: float = 1e-8
                                   ) -> List[ops.Operation]:
    """Decomposes a two-qubit operation into MS/single-qubit rotation gates.

    Args:
        q0: The first qubit being operated on.
        q1: The other qubit being operated on.
        mat: Defines the operation to apply to the pair of qubits.
        tolerance: A limit on the amount of error introduced by the
            construction.

    Returns:
        A list of operations implementing the matrix.
    """
    _, (a0, a1), (x, y, z), (b0, b1) = linalg.kak_decomposition(
        mat,
        linalg.Tolerance(atol=tolerance))
    return _kak_decomposition_to_operations(q0, q1,
                                            a0, a1, x, y, z, b0, b1,
                                            tolerance)


def _kak_decomposition_to_operations(q0: ops.QubitId,
                                     q1: ops.QubitId,
                                     a0: np.ndarray,
                                     a1: np.ndarray,
                                     x: float,
                                     y: float,
                                     z: float,
                                     b0: np.ndarray,
                                     b1: np.ndarray,
                                     tolerance: float = 1e-8
                                     ) -> List[ops.Operation]:
    """Assumes that the decomposition is canonical."""
    pre = [_do_single_on(b0, q0, tolerance), _do_single_on(b1, q1, tolerance)]
    post = [_do_single_on(a0, q0, tolerance), _do_single_on(a1, q1, tolerance)]

    return list(ops.flatten_op_tree([
        pre,
        _non_local_part(q0, q1, x, y, z, tolerance),
        post,
    ]))


def _do_single_on(u: np.ndarray, q: ops.QubitId, tolerance: float=1e-8):
    for gate in decompositions.single_qubit_matrix_to_gates(u, tolerance):
        yield gate(q)


def _parity_interaction(q0: ops.QubitId,
                        q1: ops.QubitId,
                        rads: float,
                        tolerance: float,
                        gate: Optional[ops.Gate] = None):
    """Yields a XX interaction framed by the given operation."""
    if abs(rads) < tolerance:
        return

    if gate is not None:
        g = cast(ops.Gate, gate)
        yield g.on(q0), g.on(q1)

    yield ion.MS(q0, q1) ** (-4/np.pi * rads)

    if gate is not None:
        g = protocols.inverse(gate)
        yield g.on(q0), g.on(q1)


def _non_local_part(q0: ops.QubitId,
                    q1: ops.QubitId,
                    x: float,
                    y: float,
                    z: float,
                    tolerance: float = 1e-8):
    """Yields non-local operation of KAK decomposition."""
    return[
        _parity_interaction(q0, q1, x, tolerance),
        _parity_interaction(q0, q1, y, tolerance, ops.Z ** -0.5),
        _parity_interaction(q0, q1, z, tolerance, ops.Y ** 0.5)]


def ControlledXRoot_decompose(q0: ops.QubitId,
                              q1: ops.QubitId,
                              exponent: Optional[Union[value.Symbol, float]] = None):
    """
    Decompose the gate
    ---.-----
       |
    ---X^t---
    for -1 <= t <= 1.
    :param q0: control qubit
    :param q1: target qubit
    :param exponent: the root of Pauli X that you want to perform
    :return: list of single-qubit rotation gates and MS gates that effectively perform this gate
    """
    yield ops.RotYGate(rads=-np.pi/2).on(q0)
    yield ion.MSGate(rads=exponent*np.pi/4).on(q0, q1)
    yield ops.RotXGate(rads=-exponent*np.pi/2)
    yield ops.RotXGate(rads=exponent*np.pi/2)
    yield ops.RotYGate(rads=np.pi/2)


def ControlledYRoot_decompose(q0: ops.QubitId,
                              q1: ops.QubitId,
                              exponent: Optional[Union[value.Symbol, float]] = None):
    """

    Decompose the gate
    ---.-----
       |
    ---Y^t---
    for -1 <= t <= 1.
    :param q0: control qubit
    :param q1: target qubit
    :param exponent: the root of Pauli Y that you want to perform
    :return: list of single-qubit rotation gates and MS gates that effectively perform this gate
    """
    yield protocols.inverse(ops.S).on(q1)
    yield ControlledXRoot_decompose(q0, q1, exponent)
    yield ops.S(q1)


def ControlledZRoot_decompose(q0: ops.QubitId,
                              q1: ops.QubitId,
                              exponent: Optional[Union[value.Symbol, float]] = None):
    """

    Decompose the gate
    ---.-----
       |
    ---Z^t---
    for -1 <= t <= 1.
    :param q0: control qubit
    :param q1: target qubit
    :param exponent: the root of Pauli Y that you want to perform
    :return: list of single-qubit rotation gates and MS gates that effectively perform this gate
    """
    yield protocols.inverse(ops.H).on(q1)
    yield ControlledXRoot_decompose(q0, q1, exponent)
    yield ops.H(q1)
