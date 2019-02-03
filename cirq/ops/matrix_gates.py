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

from typing import cast, Any

import numpy as np

from cirq import linalg, protocols
from cirq.ops import linear_operator, gate_features


def _phase_matrix(turns: float) -> np.ndarray:
    return np.diag([1, np.exp(2j * np.pi * turns)])


class SingleQubitMatrixGate(linear_operator.UnitaryMixin, gate_features.SingleQubitGate):
    """A 1-qubit gate defined by its matrix.

    More general than specialized classes like `ZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """
        if matrix.shape != (2, 2) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 2x2 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 1:
            raise ValueError(
                'Single-qubit gate applied to multiple qubits: {}({})'.format(
                    self, qubits))

    def __pow__(self, exponent: Any) -> 'SingleQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return SingleQubitMatrixGate(new_mat)

    def _trace_distance_bound_(self):
        vals = np.linalg.eigvals(self._matrix)
        rotation_angle = abs(np.angle(vals[0] / vals[1]))
        return rotation_angle * 1.2

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        z = _phase_matrix(phase_turns)
        phased_matrix = z.dot(self._matrix).dot(np.conj(z.T))
        return SingleQubitMatrixGate(phased_matrix)

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args),))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.SingleQubitMatrixGate({})'.format(
                _numpy_array_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


class TwoQubitMatrixGate(linear_operator.UnitaryMixin, gate_features.TwoQubitGate):
    """A 2-qubit gate defined only by its matrix.

    More general than specialized classes like `CZPowGate`, but more expensive
    and more float-error sensitive to work with (due to using
    eigendecompositions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        """
        Initializes the 2-qubit matrix gate.

        Args:
            matrix: The matrix that defines the gate.
        """
        if matrix.shape != (4, 4) or not linalg.is_unitary(matrix):
            raise ValueError('Not a 4x4 unitary matrix: {}'.format(matrix))
        self._matrix = matrix

    def validate_args(self, qubits):
        if len(qubits) != 2:
            raise ValueError(
                'Two-qubit gate not applied to two qubits: {}({})'.format(
                    self, qubits))

    def __pow__(self, exponent: Any) -> 'TwoQubitMatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b**e)
        return TwoQubitMatrixGate(new_mat)

    def _phase_by_(self, phase_turns: float, qubit_index: int):
        i = np.eye(2)
        z = _phase_matrix(phase_turns)
        z2 = np.kron(i, z) if qubit_index else np.kron(z, i)
        phased_matrix = z2.dot(self._matrix).dot(np.conj(z2.T))
        return TwoQubitMatrixGate(phased_matrix)

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def _unitary_(self) -> np.ndarray:
        return np.array(self._matrix)

    def _circuit_diagram_info_(self, args: protocols.CircuitDiagramInfoArgs
                               ) -> protocols.CircuitDiagramInfo:
        return protocols.CircuitDiagramInfo(
            wire_symbols=(_matrix_to_diagram_symbol(self._matrix, args), '#2'))

    def __hash__(self):
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((SingleQubitMatrixGate, vals))

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.alltrue(self._matrix == other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'cirq.TwoQubitMatrixGate({})'.format(
                _numpy_array_repr(self._matrix))

    def __str__(self):
        return str(self._matrix.round(3))


def _matrix_to_diagram_symbol(matrix: np.ndarray,
                              args: protocols.CircuitDiagramInfoArgs) -> str:
    if args.precision is not None:
        matrix = matrix.round(args.precision)
    result = str(matrix)
    if args.use_unicode_characters:
        lines = result.split('\n')
        for i in range(len(lines)):
            lines[i] = lines[i].replace('[[', '')
            lines[i] = lines[i].replace(' [', '')
            lines[i] = lines[i].replace(']', '')
        w = max(len(line) for line in lines)
        for i in range(len(lines)):
            lines[i] = '│' + lines[i].ljust(w) + '│'
        lines.insert(0, '┌' + ' ' * w + '┐')
        lines.append('└' + ' ' * w + '┘')
        result = '\n'.join(lines)
    return result


def _numpy_array_repr(arr: np.ndarray) -> str:
    return 'np.array({!r})'.format(arr.tolist())


def _is_power_of_two(n: int) -> bool:
    return n > 0 and n & (n-1) == 0


def _infer_n_qubits(op: linear_operator.AbstractLinearOperator) -> int:
    matrix = op.matrix()

    if matrix is None:
        raise ValueError('Cannot make gate from operator without matrix')
    if len(matrix.shape) != 2:
        raise ValueError('Cannot make gate from array of {} dimensions'
                         .format(len(matrix.shape)))
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError('Cannot make gate from rectangular matrix of shape {}'
                         .format(matrix.shape))
    if not _is_power_of_two(matrix.shape[0]):
        raise ValueError('Cannot make gate from matrix of shape {}'
                         .format(matrix.shape))

    return round(np.log2(matrix.shape[0]))


def make_gate(op: linear_operator.AbstractLinearOperator) -> raw_types.Gate:
    """Makes quantum gate from unitary operator op.

    Args:
        op: unitary linear operator to make a gate out of.

    Returns:
        1- or 2-qubit matrix gate corresponding to the given linear operator.

    Raises:
        ValueError if the given linear operator is not unitary or if it acts
        on the space of more than two qubits.
    """
    n_qubits = _infer_n_qubits(op)
    if n_qubits == 1:
        return SingleQubitMatrixGate(op.matrix())
    elif n_qubits == 2:
        return TwoQubitMatrixGate(op.matrix())
    else:
        raise ValueError('Matrix gates of {} qubits are not supported'
                         .format(n_qubits))


def forge_gate(op: linear_operator.LinearOperator) -> raw_types.Gate:
    """Makes quantum gate from unitary factor in polar decomposition of op.

    Args:
        op: linear operator to make a gate out of.

    Returns:
        1- or 2-qubit matrix gate corresponding to the given linear operator.

    Raises:
        ValueError if the given linear operator acts on the space of more than
        two qubits.
    """
    return make_gate(op.unitary_factor())
