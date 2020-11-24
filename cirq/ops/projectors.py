from typing import (Any, Dict, Iterable, Mapping, Sequence, List, Optional,
    Tuple, TYPE_CHECKING, TypeVar, Union)

import numpy as np

from cirq import linalg, value
from cirq.ops import raw_types
from cirq.qis import states
from cirq.qis import STATE_VECTOR_LIKE

if TYPE_CHECKING:
    import cirq
    from cirq import protocols

ProjKey = TypeVar('ProjKey', bound=Union[raw_types.Qid, Tuple[raw_types.Qid]])


def qid_shape_from_proj_key(proj_key):
    if isinstance(proj_key, tuple):
        return [qubit.dimension for qubit in proj_key]
    else:
        return [proj_key.dimension]


@value.value_equality
class Projector():
    """A projection matrix where you can specify the basis.

    The input is a matrix representing the basis of the space we project onto.
    The basis vectors need not be orthogonal, but they must be independent (i.e.
    the matrix has full rank).

    For example, if you want to project on |0⟩, you would provide the basis
    [[1, 0]]. To project onto |10⟩, you would provide the basis [[0, 0, 1, 0]].
    If you want to project on the space spanned by |10⟩ and |11⟩, you could
    provide the basis [[0, 0, 1, 0], [0, 0, 0, 1]].
    """

    def __init__(self,
                 projection_bases: Dict[ProjKey, Sequence[STATE_VECTOR_LIKE]],
                 enforce_orthonormal_basis: bool = False):
        """
        Args:
            projection_bases: a dictionary of Qdit tuples to a
                (p, 2**num_qubits) matrix that lists the projection vectors,
                where p is the dimension of the subspace we're projecting on. If
                you project onto a single vector, then p = 1 and thus the matrix
                reduces to a single row.
            enforce_orthonormal_basis: Whether to enfore the input basis to be
                orthogonal.

        Raises:
            ValueError: If the basis vector is empty.
        """
        self._projection_bases = {}
        for qubits, projection_basis in projection_bases.items():
            qid_shape = qid_shape_from_proj_key(qubits)
            projection_array = np.vstack([
                states.to_valid_state_vector(x, qid_shape=qid_shape)
                for x in projection_basis
            ])

            if enforce_orthonormal_basis:
                B = projection_array @ projection_array.T.conj()
                if not np.allclose(
                        B, np.eye(projection_array.shape[0]), atol=1e-6):
                    raise ValueError('The basis must be orthonormal')

            if np.linalg.matrix_rank(
                    projection_array) < projection_array.shape[0]:
                raise ValueError(
                    'Vectors in basis must be linearly independent')

            self._projection_bases[qubits] = projection_array

    def _projection_bases_(self) -> np.ndarray:
        return self._projection_bases

    def matrix(self, proj_keys: Optional[Iterable[ProjKey]] = None
              ) -> Iterable[np.ndarray]:
        proj_keys = self._projection_bases.keys(
        ) if proj_keys is None else proj_keys
        factors = []
        for proj_key in proj_keys:
            if proj_key not in self._projection_bases:
                qid_shape = qid_shape_from_proj_key(proj_key)
                factors.append(np.eye(np.prod(qid_shape)))
            else:
                # Make rows into columns
                A = self._projection_bases[proj_key].T
                # Left pseudo-inverse
                pseudoinverse = np.linalg.pinv(A)
                # Projector to the range (column space) of A
                factors.append(A @ pseudoinverse)
        return linalg.kron(*factors)

    def expectation_from_state_vector(self,
                                      state_vector: np.ndarray,
                                      qubit_map: Mapping[ProjKey, int],
                                      *,
                                      atol: float = 1e-7,
                                      check_preconditions: bool = True
                                     ) -> float:
        dims = [
            np.prod(qid_shape_from_proj_key(proj_key))
            for proj_key in qubit_map.keys()
        ]
        state_vector = state_vector.reshape(dims)

        for proj_key, i in qubit_map.items():
            if proj_key not in self._projection_bases:
                continue

            # Make rows into columns
            A = self._projection_bases[proj_key].T
            # Left pseudo-inverse
            pseudoinverse = np.linalg.pinv(A)
            # Projector to the range (column space) of A
            P = A @ pseudoinverse

            state_vector = np.tensordot(P, state_vector, axes=(1, i))

        state_vector = np.reshape(state_vector, np.prod(dims))
        return np.dot(state_vector, state_vector)

    def expectation_from_density_matrix(self,
                                        state: np.ndarray,
                                        qubit_map: Mapping[ProjKey, int],
                                        *,
                                        atol: float = 1e-7,
                                        check_preconditions: bool = True
                                       ) -> float:
        dims = [
            np.prod(qid_shape_from_proj_key(proj_key))
            for proj_key in qubit_map.keys()
        ]
        state = state.reshape(dims + dims)

        for proj_key, i in qubit_map.items():
            if proj_key not in self._projection_bases:
                continue

            # Make rows into columns
            A = self._projection_bases[proj_key].T
            # Left pseudo-inverse
            pseudoinverse = np.linalg.pinv(A)
            # Projector to the range (column space) of A
            P = A @ pseudoinverse

            state = np.tensordot(P, state, axes=(1, i))
            state = np.tensordot(state, P.T.conj(), axes=(i, 0))

        state = np.reshape(state, [np.prod(dims)] * 2)
        return np.trace(state)

    def __repr__(self) -> str:
        return f"cirq.Projector(projection_bases={self._projection_bases})"

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            # JSON requires mappings to have string keys.
            'projection_bases': list(self._projection_bases.items())
        }

    @classmethod
    def _from_json_dict_(cls, projection_bases, **kwargs):
        return cls(projection_bases=dict(projection_bases))

    def _value_equality_values_(self) -> Any:
        sorted_items = sorted(self._projection_bases.items())
        return tuple([(x[0], x[1].tobytes()) for x in sorted_items])
