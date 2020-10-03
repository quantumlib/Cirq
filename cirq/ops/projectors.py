import numpy as np
from typing import Any, Dict, Tuple, TYPE_CHECKING

from cirq import value
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


@value.value_equality
class Projector(raw_types.Gate):
    """A non-unitary gate that projects onto a single qubit."""

    def __init__(self, projector_id: int, qid_shape: int = 2):
        """
        Args:
            projector_id: An integer smaller than qid_shape that specifies the
                projector on the qubit.
            qid_shape: Specifies the dimension of the projection. The default is
                2 (qubit).

        Raises:
            ValueError: If the length of invert_mask is greater than num_qubits.
                or if the length of qid_shape doesn't equal num_qubits.
        """
        if projector_id >= qid_shape:
            raise ValueError(
                f"projector_id {projector_id} must be < qid_shape={qid_shape}")
        self._projector_id = projector_id
        self._qid_shape = qid_shape

    def _projector_id_(self) -> int:
        return self._projector_id

    def _qid_shape_(self) -> int:
        return self._qid_shape

    def _has_unitary_(self) -> bool:
        return False

    def _is_parameterized_(self) -> bool:
        return False

    def _channel_(self):
        result = np.zeros((self._qid_shape, self._qid_shape))
        result[self._projector_id][self._projector_id] = 1.0
        return (result,)

    def _has_channel_(self):
        return True

    def __repr__(self):
        return f"cirq.Projector(projector_id={self._projector_id},qid_shape={self._qid_shape})"
    
    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'projector_id': self._projector_id,
            'qid_shape': self._qid_shape,
        }
    
    @classmethod
    def _from_json_dict_(cls,
                         projector_id,
                         qid_shape):
        return cls(projector_id=projector_id, qid_shape=qid_shape)
    
    def _value_equality_values_(self) -> Any:
        return self._projector_id, self._qid_shape
