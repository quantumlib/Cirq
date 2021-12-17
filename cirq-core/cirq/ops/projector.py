# pylint: disable=wrong-or-nonexistent-copyright-notice
import itertools
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Union,
)

import numpy as np
from scipy.sparse import csr_matrix

from cirq import value
from cirq.ops import raw_types


def _check_qids_dimension(qids):
    """A utility to check that we only have Qubits."""
    for qid in qids:
        if qid.dimension != 2:
            raise ValueError(f"Only qubits are supported, but {qid} has dimension {qid.dimension}")


@value.value_equality(approximate=True)
class ProjectorString:
    def __init__(
        self,
        projector_dict: Dict[raw_types.Qid, int],
        coefficient: Union[int, float, complex] = 1,
    ):
        """Contructor for ProjectorString

        Args:
            projector_dict: A python dictionary mapping from cirq.Qid to integers. A key value pair
                represents the desired computational basis state for that qubit.
            coefficient: Initial scalar coefficient. Defaults to 1.
        """
        _check_qids_dimension(projector_dict.keys())
        self._projector_dict = projector_dict
        self._coefficient = complex(coefficient)

    @property
    def projector_dict(self) -> Dict[raw_types.Qid, int]:
        return self._projector_dict

    @property
    def coefficient(self) -> complex:
        return self._coefficient

    def matrix(self, projector_qids: Optional[Iterable[raw_types.Qid]] = None) -> csr_matrix:
        """Returns the matrix of self in computational basis of qubits.

        Args:
            projector_qids: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the ProjectorString is to
                be computed. Qbits absent from self.qubits are acted on by
                the identity. Defaults to the qubits of the projector_dict.

        Returns:
            A sparse matrix that is the projection in the specified basis.
        """
        projector_qids = self._projector_dict.keys() if projector_qids is None else projector_qids
        _check_qids_dimension(projector_qids)
        idx_to_keep = [
            [self._projector_dict[qid]] if qid in self._projector_dict else [0, 1]
            for qid in projector_qids
        ]

        total_d = np.prod([qid.dimension for qid in projector_qids], dtype=np.int64)

        ones_idx = []
        for idx in itertools.product(*idx_to_keep):
            d = total_d
            kron_idx = 0
            for i, qid in zip(idx, projector_qids):
                d //= qid.dimension
                kron_idx += i * d
            ones_idx.append(kron_idx)

        return csr_matrix(
            ([self._coefficient] * len(ones_idx), (ones_idx, ones_idx)), shape=(total_d, total_d)
        )

    def _get_idx_to_keep(self, qid_map: Mapping[raw_types.Qid, int]):
        num_qubits = len(qid_map)
        idx_to_keep: List[Any] = [slice(0, 2)] * num_qubits
        for q in self.projector_dict.keys():
            idx_to_keep[qid_map[q]] = self.projector_dict[q]
        return tuple(idx_to_keep)

    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
    ) -> complex:
        """Expectation of the projection from a state vector.

        Computes the expectation value of this ProjectorString on the provided state vector.

        Args:
            state_vector: An array representing a valid state vector.
            qid_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
        _check_qids_dimension(qid_map.keys())
        num_qubits = len(qid_map)
        index = self._get_idx_to_keep(qid_map)
        return self._coefficient * np.sum(
            np.abs(np.reshape(state_vector, (2,) * num_qubits)[index]) ** 2
        )

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
    ) -> complex:
        """Expectation of the projection from a density matrix.

        Computes the expectation value of this ProjectorString on the provided state.

        Args:
            state: An array representing a valid  density matrix.
            qid_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.

        Returns:
            The expectation value of the input state.
        """
        _check_qids_dimension(qid_map.keys())
        num_qubits = len(qid_map)
        index = self._get_idx_to_keep(qid_map) * 2
        result = np.reshape(state, (2,) * (2 * num_qubits))[index]
        while any(result.shape):
            result = np.trace(result, axis1=0, axis2=len(result.shape) // 2)
        return self._coefficient * result

    def __repr__(self) -> str:
        return (
            f"cirq.ProjectorString(projector_dict={self._projector_dict},"
            + f"coefficient={self._coefficient})"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'projector_dict': list(self._projector_dict.items()),
            'coefficient': self._coefficient,
        }

    @classmethod
    def _from_json_dict_(cls, projector_dict, coefficient, **kwargs):
        return cls(projector_dict=dict(projector_dict), coefficient=coefficient)

    def _value_equality_values_(self) -> Any:
        projector_dict = sorted(self._projector_dict.items())
        return (tuple(projector_dict), self._coefficient)
