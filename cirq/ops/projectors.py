from typing import Any, Dict, Iterable, List, Tuple, TYPE_CHECKING, Union

import numpy as np

from cirq import value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class Projector(raw_types.Gate):
    """A non-unitary gate that projects onto one or more qubits.

    The input is a matrix representing the basis of the space we project onto.
    The basis vectors need not be orthogonal, but they must be independent (i.e.
    the matrix has full rank).

    For example, if you want to project on |0>, you would provide the basis
    [[1, 0]]. To project onto |10>, you would provide the basis [[0, 0, 1, 0]].
    If you want to project on the space spanned by |10> and |11>, you could
    provide the basis [[0, 0, 1, 0], [0, 0, 0, 1]].
    """

    def __init__(self,
                 projection_basis: Union[List[List[float]], np.ndarray],
                 qid_shape: Tuple[int, ...] = (2,),
                 enfore_orthogonal_basis: bool = False):
        """
        Args:
            projection_basis: a (p, 2**num_qubits) matrix that lists the
                projection vectors, where p is the dimension of the subspace
                we're projecting on. If you project onto a single vector, then
                p = 1 and thus the matrix reduces to a single row.
            qid_shape: Specifies the dimension of the projection. The default is
                2 (qubit).
            enfore_orthogonal_basis: Whether to enfore the input basis to be
                orthogonal.

        Raises:
            ValueError: If the basis vector is empty.
        """
        projection_array = np.asarray(projection_basis)

        if len(projection_array.shape) != 2:
            raise ValueError('The input projection_basis must be a 2D array')

        if enfore_orthogonal_basis:
            B = projection_array @ np.transpose(np.conjugate(projection_array))
            if not np.allclose(B, np.eye(projection_array.shape[0])):
                raise ValueError('The basis must be orthogonal')

        if np.linalg.matrix_rank(projection_array) < projection_array.shape[0]:
            raise ValueError('Vectors in basis must be linearly independent')

        if np.prod(qid_shape) != projection_array.shape[1]:
            raise ValueError(
                "Invalid shape " +
                f"{np.array(projection_array).shape} for qid_shape {qid_shape}")
        self._projection_basis = projection_array
        self._qid_shape = qid_shape

    def _projection_basis_(self) -> np.ndarray:
        return self._projection_basis

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return False

    def _is_parameterized_(self) -> bool:
        return False

    def _channel_(self) -> Iterable[np.ndarray]:
        A = self._projection_basis
        AH = np.transpose(np.conjugate(A))
        pseudoinverse = AH @ np.linalg.inv(A @ AH)
        return (pseudoinverse @ A,)

    def _has_channel_(self) -> bool:
        return True

    def __repr__(self) -> str:
        return ("cirq.Projector(projection_basis=" +
                f"{self._projection_basis.tolist()})," +
                f"qid_shape={self._qid_shape})")

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'projection_basis': self._projection_basis.tolist(),
            'qid_shape': self._qid_shape,
        }

    def _value_equality_values_(self) -> Any:
        return self._projection_basis.tolist(), self._qid_shape
