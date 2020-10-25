from typing import Any, Dict, List, Tuple, TYPE_CHECKING

import numpy as np

from cirq import value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class Projector(raw_types.Gate):
    """A non-unitary gate that projects onto a single qubit."""

    def __init__(self,
                 projection_basis: List[List[float]],
                 qid_shape: Tuple[int, ...] = (2,)):
        """
        Args:
            projection_basis: a (2**num_qubits, p) matrix that lists the
                proection vectors.
            qid_shape: Specifies the dimension of the projection. The default is
                2 (qubit).

        Raises:
            ValueError: If the basis vector is empty.
        """
        if np.prod(qid_shape) != np.array(projection_basis).shape[1]:
            raise ValueError(f"Invalid shape {np.array(projection_basis).shape} for qid_shape {qid_shape}")
        self._projection_basis = projection_basis
        self._qid_shape = qid_shape

    def _projection_basis_(self):
        return self._projection_basis

    def _qid_shape_(self) -> Tuple[int, ...]:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return False

    def _is_parameterized_(self) -> bool:
        return False

    def _channel_(self):
        A = np.array(self._projection_basis)
        AH = np.transpose(np.conjugate(A))
        result = AH * np.linalg.inv(np.matmul(A, AH)) * A
        return (result,)

    def _has_channel_(self):
        return True

    def __repr__(self):
        return (f"cirq.Projector(projection_basis={self._projection_basis})," +
                f"qid_shape={self._qid_shape})")

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'projection_basis': self._projection_basis,
            'qid_shape': self._qid_shape,
        }

    def _value_equality_values_(self) -> Any:
        return self._projection_basis, self._qid_shape
