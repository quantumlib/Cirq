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

"""Quantum gates defined by a matrix."""

from typing import Optional

import numpy as np

from cirq import linalg
from cirq.ops import gate_features, raw_types


def _phase_matrix(turns: float) -> np.ndarray:
    return np.diag([1, np.exp(2j * np.pi * turns)])


class SingleQubitMatrixGate(raw_types.Gate,
                            gate_features.KnownMatrix,
                            gate_features.PhaseableEffect,
                            gate_features.ExtrapolatableEffect,
                            gate_features.BoundedEffect):
    """A 1-qubit gate defined by its matrix.

    More general than specialized classes like ZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: Optional[np.ndarray]) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate. Child classes can instead
                instead implement the matrix method and pass in None.
        """
        if matrix is not None and (matrix.shape != (2, 2) or
                                   not linalg.is_unitary(matrix)):
            raise ValueError('Not a 2x2 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 1:
            raise ValueError(
                'Single-qubit gate applied to multiple qubits: {}({})'.format(
                    self, qubits))

    def extrapolate_effect(self, factor: float):
        new_mat = linalg.map_eigenvalues(self.matrix(), lambda e: e**factor)
        return SingleQubitMatrixGate(new_mat)

    def trace_distance_bound(self):
        vals = np.linalg.eigvals(self.matrix())
        rotation_angle = abs(np.angle(vals[0] / vals[1]))
        return rotation_angle * 1.2

    def phase_by(self, phase_turns: float, qubit_index: int):
        z = _phase_matrix(phase_turns)
        phased_matrix = z.dot(self.matrix()).dot(np.conj(z.T))
        return SingleQubitMatrixGate(phased_matrix)

    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            raise NotImplementedError(
                'Children of {} must either provide a '
                'matrix to the init method or else implement the matrix '
                'method.'.format(type(self)))
        return self._matrix

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self.matrix()))
        return hash((SingleQubitMatrixGate, vals))

    def approx_eq(self, other, ignore_global_phase=True):
        if not isinstance(other, type(self)):
            return NotImplemented
        cmp = (linalg.allclose_up_to_global_phase if ignore_global_phase
               else np.allclose)
        return cmp(self.matrix(), other.matrix())

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self.matrix() == other.matrix())

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'SingleQubitMatrixGate({})'.format(repr(self.matrix()))

    def __str__(self):
        return str(self.matrix().round(3))


class TwoQubitMatrixGate(raw_types.Gate,
                         gate_features.KnownMatrix,
                         gate_features.PhaseableEffect,
                         gate_features.ExtrapolatableEffect):
    """A 2-qubit gate defined only by its matrix.

    More general than specialized classes like CZGate, but more expensive and
    more float-error sensitive to work with (due to using eigendecompositions).
    """

    def __init__(self, matrix: Optional[np.ndarray]) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate. Child classes can instead
                instead implement the matrix method and pass in None.
        """

        if matrix is not None and (matrix.shape != (4, 4) or
                                   not linalg.is_unitary(matrix)):
            raise ValueError('Not a 4x4 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 2:
            raise ValueError(
                'Two-qubit gate not applied to two qubits: {}({})'.format(
                    self, qubits))

    def extrapolate_effect(self, factor: float):
        new_mat = linalg.map_eigenvalues(self.matrix(), lambda e: e**factor)
        return TwoQubitMatrixGate(new_mat)

    def phase_by(self, phase_turns: float, qubit_index: int):
        i = np.eye(2)
        z = _phase_matrix(phase_turns)
        z2 = np.kron(z, i) if qubit_index else np.kron(i, z)
        phased_matrix = z2.dot(self.matrix()).dot(np.conj(z2.T))
        return TwoQubitMatrixGate(phased_matrix)

    def approx_eq(self, other, ignore_global_phase=True):
        if not isinstance(other, type(self)):
            return NotImplemented
        cmp = (linalg.allclose_up_to_global_phase if ignore_global_phase
               else np.allclose)
        return cmp(self.matrix(), other.matrix())

    def matrix(self):
        if self._matrix is None:
            raise NotImplementedError(
                'Children of {} must either provide a '
                'matrix to the init method or else implement the matrix '
                'method.'.format(type(self)))
        return self._matrix

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self.matrix()))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self.matrix() == other.matrix())

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'TwoQubitMatrixGate({})'.format(repr(self.matrix()))

    def __str__(self):
        return str(self.matrix().round(3))
