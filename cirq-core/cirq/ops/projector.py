import itertools
from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Union,
)

import numpy as np
from scipy.sparse import coo_matrix

from cirq import value
from cirq.ops import raw_types


def get_sorted_qids(qid_map: Mapping[raw_types.Qid, int]):
    sorted_pairs = sorted([(i, qid) for qid, i in qid_map.items()])
    return [x[1] for x in sorted_pairs]


def check_qids_dimension(qids):
    for qid in qids:
        if qid.dimension != 2:
            raise ValueError(f"Only qubits are supported, but {qid} has dimension {qid.dimension}")


@value.value_equality
class ProjectorString:
    def __init__(
        self,
        projector_dict: Dict[raw_types.Qid, int],
        coefficient: Union[int, float, complex] = 1,
    ):
        """Contructor for ProjectorString

        Args:
            projector_dict: a dictionary of Qdit tuples to an integer specifying which vector to
                project on.
        """
        check_qids_dimension(projector_dict.keys())
        self._projector_dict = projector_dict
        self._coefficient = complex(coefficient)

    @property
    def projector_dict(self) -> Dict[raw_types.Qid, int]:
        return self._projector_dict

    @property
    def coefficient(self) -> complex:
        return self._coefficient

    def matrix(self, projector_qids: Optional[Iterable[raw_types.Qid]] = None) -> coo_matrix:
        projector_qids = self._projector_dict.keys() if projector_qids is None else projector_qids
        check_qids_dimension(projector_qids)
        idx_to_keep = self._get_idx_to_keep(projector_qids)

        total_d = np.prod([qid.dimension for qid in projector_qids])

        ones_idx = []
        for idx in idx_to_keep:
            assert len(idx) == len(list(projector_qids))
            d = total_d
            kron_idx = 0
            for i, qid in zip(idx, projector_qids):
                d //= qid.dimension
                kron_idx += i * d
            ones_idx.append(kron_idx)

        return coo_matrix(([1.0] * len(ones_idx), (ones_idx, ones_idx)), shape=(total_d, total_d))

    def _get_idx_to_keep(self, sorted_qid: Iterable[raw_types.Qid]):
        idx_to_keep = []
        for qid in sorted_qid:
            if qid in self._projector_dict:
                idx_to_keep.append([self._projector_dict[qid]])
            else:
                idx_to_keep.append(list(range(qid.dimension)))
        return itertools.product(*idx_to_keep)

    def _check_all_qids_present(self, qid_map: Mapping[raw_types.Qid, int]):
        for qid in self._projector_dict.keys():
            if qid not in qid_map:
                raise ValueError(f"Missing qid: {qid}")

    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> complex:
        check_qids_dimension(qid_map.keys())
        self._check_all_qids_present(qid_map)
        sorted_qid = get_sorted_qids(qid_map)
        state_vector = state_vector.reshape([qid.dimension for qid in sorted_qid]).copy()
        idx_to_keep = self._get_idx_to_keep(sorted_qid)
        return self._coefficient * sum(np.abs(state_vector[idx]) ** 2 for idx in idx_to_keep)

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> complex:
        check_qids_dimension(qid_map)
        self._check_all_qids_present(qid_map)
        sorted_qid = get_sorted_qids(qid_map)
        state = state.reshape([qid.dimension for qid in sorted_qid] * 2).copy()
        idx_to_keep = self._get_idx_to_keep(sorted_qid)
        return self._coefficient * sum(np.abs(state[idx + idx]) ** 2 for idx in idx_to_keep)

    def __repr__(self) -> str:
        return (
            f"cirq.ProjectorString(projector_dict={self._projector_dict},"
            + f"coefficient={self._coefficient})"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'projector_dict': list(self._projector_dict.items()),
            'coefficient': self._coefficient,
        }

    @classmethod
    def _from_json_dict_(cls, projector_dict, coefficient, **kwargs):
        return cls(projector_dict=dict(projector_dict), coefficient=coefficient)

    def _value_equality_values_(self) -> Any:
        projector_dict = sorted(self._projector_dict.items())
        return (tuple(projector_dict), self._coefficient)
