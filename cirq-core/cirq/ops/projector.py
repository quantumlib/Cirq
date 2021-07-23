from collections import defaultdict
import itertools
import numbers
from typing import (
    Any,
    DefaultDict,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
<<<<<<< HEAD
    Tuple,
=======
>>>>>>> projector_strings
    Union,
)

import numpy as np
from scipy.sparse import coo_matrix

from cirq import value
from cirq.ops import raw_types


def check_qids_dimension(qids):
    """A utility to check that we only have Qubits."""
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
            projector_dict: a dictionary of Qbit to an integer specifying which vector to project
                onto.
            coefficient: Initial scalar coefficient. Defaults to 1.
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
        """Returns the matrix of self in computational basis of qubits.

        Args:
            projector_qids: Ordered collection of qubits that determine the subspace
                in which the matrix representation of the ProjectorString is to
                be computed. Qbits absent from self.qubits are acted on by
                the identity. Defaults to the qubits of the projector_dict.
        """
        projector_qids = self._projector_dict.keys() if projector_qids is None else projector_qids
        check_qids_dimension(projector_qids)
        idx_to_keep = []
        for qid in projector_qids:
            if qid in self._projector_dict:
                idx_to_keep.append([self._projector_dict[qid]])
            else:
                idx_to_keep.append(list(range(qid.dimension)))

        total_d = np.prod([qid.dimension for qid in projector_qids])

        ones_idx = []
        for idx in itertools.product(*idx_to_keep):
            assert len(idx) == len(list(projector_qids))
            d = total_d
            kron_idx = 0
            for i, qid in zip(idx, projector_qids):
                d //= qid.dimension
                kron_idx += i * d
            ones_idx.append(kron_idx)

        return coo_matrix(([1.0] * len(ones_idx), (ones_idx, ones_idx)), shape=(total_d, total_d))

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

        Projects the state vector onto the projector_dict and computes the expectation of the
        measurement.

        Args:
            state_vector: An array representing a valid state vector.
            qubit_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.
        Returns:
            The expectation value of the input state.
        """
        check_qids_dimension(qid_map.keys())
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

        Projects the density matrix onto the projector_dict and computes the expectation of the
        measurement.

        Args:
            state: An array representing a valid  density matrix.
            qubit_map: A map from all qubits used in this ProjectorString to the
                indices of the qubits that `state_vector` is defined over.
        Returns:
            The expectation value of the input state.
        """
        check_qids_dimension(qid_map.keys())
        num_qubits = len(qid_map)
        index = self._get_idx_to_keep(qid_map) * 2
        return self._coefficient * np.sum(
            np.abs(np.reshape(state, (2,) * (2 * num_qubits))[index]) ** 2
        )

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


def _projector_string_from_projector_dict(projector_dict):
    return ProjectorString(dict(projector_dict))


@value.value_equality(approximate=True)
class ProjectorSum:
    def __init__(self, linear_dict=None):
        self._linear_dict = linear_dict if linear_dict is not None else {}

    def _value_equality_values_(self):
        return self._linear_dict

    def _json_dict_(self) -> Dict[str, Any]:
        linear_dict = []
        for projector_dict, scalar in dict(self._linear_dict).items():
            key = [[k, v] for k, v in dict(projector_dict).items()]
            linear_dict.append([key, scalar])
        return {
            'cirq_type': self.__class__.__name__,
            'linear_dict': linear_dict,
        }

    @classmethod
    def _from_json_dict_(cls, linear_dict, **kwargs):
        converted_dict = {}
        for projector_string in linear_dict:
            projector_dict = {x[0]: x[1] for x in projector_string[0]}
            scalar = projector_string[1]
            key = frozenset(projector_dict.items())
            converted_dict[key] = scalar
        return cls(linear_dict=value.LinearDict(converted_dict))

    @classmethod
    def from_projector_strings(
        cls, terms: Union[ProjectorString, List[ProjectorString]]
    ) -> 'ProjectorSum':
        if isinstance(terms, ProjectorString):
            terms = [terms]
        termdict: DefaultDict[FrozenSet[Tuple[raw_types.Qid, int]], value.Scalar] = defaultdict(
            lambda: 0.0
        )
        for pstring in terms:
            key = frozenset(pstring._projector_dict.items())
            termdict[key] += 1.0
        return cls(linear_dict=value.LinearDict(termdict))

    def copy(self) -> 'ProjectorSum':
        return ProjectorSum(self._linear_dict.copy())

    def matrix(self, projector_qids: Optional[Iterable[raw_types.Qid]] = None) -> coo_matrix:
        return sum(
            coeff * _projector_string_from_projector_dict(vec).matrix(projector_qids)
            for vec, coeff in self._linear_dict.items()
        )

    def expectation_from_state_vector(
        self,
        state_vector: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        return sum(
            coeff
            * _projector_string_from_projector_dict(vec).expectation_from_state_vector(
                state_vector, qid_map
            )
            for vec, coeff in self._linear_dict.items()
        )

    def expectation_from_density_matrix(
        self,
        state: np.ndarray,
        qid_map: Mapping[raw_types.Qid, int],
        *,
        atol: float = 1e-7,
        check_preconditions: bool = True,
    ) -> float:
        return sum(
            coeff
            * _projector_string_from_projector_dict(vec).expectation_from_density_matrix(
                state, qid_map
            )
            for vec, coeff in self._linear_dict.items()
        )

    def __iadd__(self, other: 'ProjectorSum'):
        result = self.copy()
        result._linear_dict += other._linear_dict
        return result

    def __add__(self, other: 'ProjectorSum'):
        result = self.copy()
        result += other
        return result

    def __imul__(self, other: numbers.Complex):
        result = self.copy()
        result._linear_dict *= other
        return result

    def __rmul__(self, other: numbers.Complex):
        result = self.copy()
        result *= other
        return result
