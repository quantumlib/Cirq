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

from typing import Any, cast, Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import numpy as np

from cirq import linalg, protocols
from cirq._compat import proper_repr
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class MatrixGate(raw_types.Gate):
    """A unitary qubit or qudit gate defined entirely by its matrix."""

    def __init__(
        self,
        matrix: np.ndarray,
        *,
        name: str = None,
        qid_shape: Optional[Iterable[int]] = None,
        unitary_check_rtol: float = 1e-5,
        unitary_check_atol: float = 1e-8,
    ) -> None:
        """Initializes a matrix gate.

        Args:
            matrix: The matrix that defines the gate.
            name: The optional name of the gate to be displayed.
            qid_shape: The shape of state tensor that the matrix applies to.
                If not specified, this value is inferred by assuming that the
                matrix is supposed to apply to qubits.
            unitary_check_rtol: The relative tolerance for checking whether the supplied matrix
                is unitary. See `cirq.is_unitary`.
            unitary_check_atol: The absolute tolerance for checking whether the supplied matrix
                is unitary. See `cirq.is_unitary`.

        Raises:
            ValueError: If the matrix is not a square numpy array, if the matrix does not match
                the `qid_shape`, if `qid_shape` is not supplied and the matrix dimension is
                not a power of 2, or if the matrix not unitary (to the supplied precisions).
        """
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError('`matrix` must be a square 2d numpy array.')

        if qid_shape is None:
            n = int(np.round(np.log2(matrix.shape[0] or 1)))
            if 2 ** n != matrix.shape[0]:
                raise ValueError(
                    f'Matrix width ({matrix.shape[0]}) is not a power of 2 and '
                    f'qid_shape is not specified.'
                )
            qid_shape = (2,) * n

        self._matrix = matrix
        self._qid_shape = tuple(qid_shape)
        self._name = name
        m = int(np.prod(self._qid_shape, dtype=np.int64))
        if self._matrix.shape != (m, m):
            raise ValueError(
                'Wrong matrix shape for qid_shape.\n'
                f'Matrix shape: {self._matrix.shape}\n'
                f'qid_shape: {self._qid_shape}\n'
            )

        if not linalg.is_unitary(matrix, rtol=unitary_check_rtol, atol=unitary_check_atol):
            raise ValueError(f'Not a unitary matrix: {self._matrix}')

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'matrix': self._matrix.tolist(),
            'qid_shape': self._qid_shape,
        }

    @classmethod
    def _from_json_dict_(cls, matrix, qid_shape, **kwargs):
        return cls(matrix=np.array(matrix), qid_shape=qid_shape)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def __pow__(self, exponent: Any) -> 'MatrixGate':
        if not isinstance(exponent, (int, float)):
            return NotImplemented
        e = cast(float, exponent)
        new_mat = linalg.map_eigenvalues(self._matrix, lambda b: b ** e)
        return MatrixGate(new_mat, qid_shape=self._qid_shape)

    def _phase_by_(self, phase_turns: float, qubit_index: int) -> 'MatrixGate':
        if not isinstance(phase_turns, (int, float)):
            return NotImplemented
        if self._qid_shape[qubit_index] != 2:
            return NotImplemented
        result = np.copy(self._matrix).reshape(self._qid_shape * 2)

        p = np.exp(2j * np.pi * phase_turns)
        i = qubit_index
        j = qubit_index + len(self._qid_shape)
        result[linalg.slice_for_qubits_equal_to([i], 1)] *= p
        result[linalg.slice_for_qubits_equal_to([j], 1)] *= np.conj(p)
        return MatrixGate(matrix=result.reshape(self._matrix.shape), qid_shape=self._qid_shape)

    def _has_unitary_(self) -> bool:
        return True

    def _unitary_(self) -> np.ndarray:
        return np.copy(self._matrix)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        n_qubits = len(self._qid_shape)
        if self._name is not None:
            symbols = (
                [self._name]
                if n_qubits == 1
                else [f'{self._name}[{i+1}]' for i in range(0, n_qubits)]
            )
            return protocols.CircuitDiagramInfo(wire_symbols=symbols)
        main = _matrix_to_diagram_symbol(self._matrix, args)
        rest = [f'#{i+1}' for i in range(1, n_qubits)]
        return protocols.CircuitDiagramInfo(wire_symbols=[main, *rest])

    def __hash__(self) -> int:
        vals = tuple(v for _, v in np.ndenumerate(self._matrix))
        return hash((MatrixGate, vals))

    def _approx_eq_(self, other: Any, atol) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return np.allclose(self._matrix, other._matrix, rtol=0, atol=atol)

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self._qid_shape == other._qid_shape and np.array_equal(self._matrix, other._matrix)

    def __ne__(self, other):
        return not self == other

    def __repr__(self) -> str:
        if all(e == 2 for e in self._qid_shape):
            return f'cirq.MatrixGate({proper_repr(self._matrix)})'
        return f'cirq.MatrixGate({proper_repr(self._matrix)}, qid_shape={self._qid_shape})'

    def __str__(self) -> str:
        return str(self._matrix.round(3))


def _matrix_to_diagram_symbol(matrix: np.ndarray, args: 'protocols.CircuitDiagramInfoArgs') -> str:
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
