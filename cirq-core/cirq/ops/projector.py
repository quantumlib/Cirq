from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
)

import numpy as np

from cirq import linalg, value
from cirq.ops import raw_types


def get_dims_from_qid_map(qid_map: Mapping[raw_types.Qid, int]):
    dims = sorted([(i, qid.dimension) for qid, i in qid_map.items()])
    return [x[1] for x in dims]


def get_qid_indices(qid_map: Mapping[raw_types.Qid, int], projector_qid: raw_types.Qid):
    qid = projector_qid
    if qid not in qid_map:
        raise ValueError(f"Missing qid: {qid}")
    return [qid_map[qid]]


@value.value_equality
class ProjectorString:
    def __init__(
        self,
        projector_dict: Dict[raw_types.Qid, int],
    ):
        """Contructor for ProjectorString

        Args:
            projector_dict: a dictionary of Qdit tuples to an integer specifying which vector to
                project on.
        """
        self._projector_dict = projector_dict

    def _projector_dict_(self) -> Dict[raw_types.Qid, int]:
        return self._projector_dict

    def _op_matrix(self, projector_qid: raw_types.Qid) -> np.ndarray:
        # TODO(tonybruguier): Consider using scipy.sparsematrix instead of a dense one.
        op_matrix = np.zeros((projector_qid.dimension,)*2)
        i = self._projector_dict[projector_qid]
        op_matrix[i][i] = 1.0
        return op_matrix

    def matrix(
        self, projector_qids: Optional[Iterable[raw_types.Qid]] = None
    ) -> Iterable[np.ndarray]:
        projector_qids = self._projector_dict.keys() if projector_qids is None else projector_qids
        factors = []
        for projector_qid in projector_qids:
            if projector_qid not in self._projector_dict.keys():
                qid_shape = [projector_qid.dimension]
                factors.append(np.eye(np.prod(qid_shape)))
            else:
                factors.append(self._op_matrix(projector_qid))
        return linalg.kron(*factors)

    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        dims = get_dims_from_qid_map(qid_map)
        state_vector = state_vector.reshape(dims)

        for projector_qid in self._projector_dict.keys():
            idx = get_qid_indices(qid_map, projector_qid)
            op_dims = [projector_qid.dimension]
            nr = len(idx)

            P = self._op_matrix(projector_qid)
            P = np.reshape(P, op_dims * 2)

            state_vector = np.tensordot(P, state_vector, axes=(range(nr, 2 * nr), idx))
            state_vector = np.moveaxis(state_vector, range(nr), idx)

        state_vector = np.reshape(state_vector, np.prod(dims))
        return np.dot(state_vector, state_vector.conj())

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        dims = get_dims_from_qid_map(qid_map)
        state = state.reshape(dims * 2)

        for projector_qid in self._projector_dict.keys():
            idx = get_qid_indices(qid_map, projector_qid)
            op_dims = [projector_qid.dimension]
            nr = len(idx)

            P = self._op_matrix(projector_qid)
            P = np.reshape(P, op_dims * 2)

            state = np.tensordot(P, state, axes=(range(nr, 2 * nr), idx))
            state = np.moveaxis(state, range(nr), idx)
            state = np.tensordot(state, P.T.conj(), axes=([len(dims) + i for i in idx], range(nr)))
            state = np.moveaxis(state, range(-nr, 0), [len(dims) + i for i in idx])

        state = np.reshape(state, [np.prod(dims)] * 2)
        return np.trace(state)

    def __repr__(self) -> str:
        return f"cirq.ProjectorString(projector_dict={self._projector_dict})"

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'projector_dict': list(self._projector_dict.items()),
        }

    @classmethod
    def _from_json_dict_(cls, projector_dict, **kwargs):
        return cls(projector_dict=dict(projector_dict))

    def _value_equality_values_(self) -> Any:
        projector_dict = sorted(self._projector_dict.items())
        return tuple(projector_dict)
