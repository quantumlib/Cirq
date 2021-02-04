from typing import (
    Any,
    Dict,
    Iterable,
    Mapping,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
    Union,
)

import numpy as np

from cirq import linalg, value
from cirq.ops import raw_types
from cirq.qis import states
from cirq.qis import STATE_VECTOR_LIKE

if TYPE_CHECKING:
    import cirq
    from cirq import protocols

KetBraKey = TypeVar('KetBraKey', bound=Union[raw_types.Qid, Tuple[raw_types.Qid]])


def qid_shape_from_ket_bra_key(ket_bra_key: KetBraKey):
    if isinstance(ket_bra_key, tuple):
        return [qid.dimension for qid in ket_bra_key]
    else:
        return [ket_bra_key.dimension]


def get_dims_from_qid_map(qid_map: Mapping[raw_types.Qid, int]):
    dims = sorted([(i, qid.dimension) for qid, i in qid_map.items()])
    return [x[1] for x in dims]


def get_qid_indices(qid_map: Mapping[raw_types.Qid, int], ket_bra_key: KetBraKey):
    if isinstance(ket_bra_key, raw_types.Qid):
        qid = ket_bra_key
        if qid not in qid_map:
            raise ValueError(f"Missing qid: {qid}")
        return [qid_map[qid]]
    else:
        idx = []
        for qid in ket_bra_key:
            if qid not in qid_map:
                raise ValueError(f"Missing qid: {qid}")
            idx.append(qid_map[qid])
        return idx


@value.value_equality
class KetBra:
    def __init__(
        self,
        ket: Optional[STATE_VECTOR_LIKE] = None,
        bra: Optional[STATE_VECTOR_LIKE] = None,
        ket_bra_list: Optional[List[Tuple[STATE_VECTOR_LIKE, STATE_VECTOR_LIKE]]] = None,
    ):

        if ket_bra_list is not None:
            self.ket_bra_list = ket_bra_list
        else:
            self.ket_bra_list = []
        if ket is not None and bra is not None:
            self.ket_bra_list.append(
                (
                    ket,
                    bra,
                )
            )

    def KetBra(self, ket: STATE_VECTOR_LIKE, bra: STATE_VECTOR_LIKE):
        self.ket_bra_list.append(
            (
                ket,
                bra,
            )
        )
        return self

    def __iter__(self):
        return iter(self.ket_bra_list)

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'ket_bra_list': [list(ket_bra) for ket_bra in self.ket_bra_list],
        }

    @classmethod
    def _from_json_dict_(cls, ket_bra_list, **kwargs):
        ket_bra_list = [tuple(ket_bra) for ket_bra in ket_bra_list]
        return cls(ket_bra_list=ket_bra_list)

    def _value_equality_values_(self) -> Any:
        return tuple(self.ket_bra_list)


@value.value_equality
class KetBraSum:
    """A generic operation specified as a list of |ket><bra|.

    The matrix is specified as a dictionary of qubits to IDs. The IDs are
    specified as |ket><bra|. Note that the matrix is not necessarily unitary,
    nor necessarily a idempotent (a projection). It is up to the caller to
    ensure that it makes sense to use the object.
    """

    def __init__(
        self,
        ket_bra_dict: Dict[KetBraKey, KetBra],
    ):
        """
        Args:
            ket_bra_dict: a dictionary of Qdit tuples to a list of |ket><bra|
                which express the operation to apply
        """
        self._ket_bra_dict = ket_bra_dict

    def _ket_bra_dict_(self) -> Dict[KetBraKey, KetBra]:
        return self._ket_bra_dict

    def _op_matrix(self, ket_bra_key: KetBraKey) -> np.ndarray:
        # TODO(tonybruguier): Speed up computation when the ket and bra are
        # encoded as integers. This probably means not calling this function at
        # all, as encoding a matrix with a single non-zero entry is not
        # efficient.
        qid_shape = qid_shape_from_ket_bra_key(ket_bra_key)

        P = 0
        for ket_bra in self._ket_bra_dict[ket_bra_key]:
            ket = states.to_valid_state_vector(ket_bra[0], qid_shape=qid_shape)
            bra = states.to_valid_state_vector(ket_bra[1], qid_shape=qid_shape).conj()
            P = P + np.einsum('i,j->ij', ket, bra)

        return P

    def matrix(self, ket_bra_keys: Optional[Iterable[KetBraKey]] = None) -> Iterable[np.ndarray]:
        ket_bra_keys = self._ket_bra_dict.keys() if ket_bra_keys is None else ket_bra_keys
        factors = []
        for ket_bra_key in ket_bra_keys:
            if ket_bra_key not in self._ket_bra_dict.keys():
                qid_shape = qid_shape_from_ket_bra_key(ket_bra_key)
                factors.append(np.eye(np.prod(qid_shape)))
            else:
                factors.append(self._op_matrix(ket_bra_key))
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

        for ket_bra_key in self._ket_bra_dict.keys():
            idx = get_qid_indices(qid_map, ket_bra_key)
            op_dims = qid_shape_from_ket_bra_key(ket_bra_key)
            nr = len(idx)

            P = self._op_matrix(ket_bra_key)
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

        for ket_bra_key in self._ket_bra_dict.keys():
            idx = get_qid_indices(qid_map, ket_bra_key)
            op_dims = qid_shape_from_ket_bra_key(ket_bra_key)
            nr = len(idx)

            P = self._op_matrix(ket_bra_key)
            P = np.reshape(P, op_dims * 2)

            state = np.tensordot(P, state, axes=(range(nr, 2 * nr), idx))
            state = np.moveaxis(state, range(nr), idx)
            state = np.tensordot(state, P.T.conj(), axes=([len(dims) + i for i in idx], range(nr)))
            state = np.moveaxis(state, range(-nr, 0), [len(dims) + i for i in idx])

        state = np.reshape(state, [np.prod(dims)] * 2)
        return np.trace(state)

    def __repr__(self) -> str:
        return f"cirq.KetBraSum(ket_bra_dict={self._ket_bra_dict})"

    def _json_dict_(self) -> Dict[str, Any]:
        encoded_dict = {k: [(t[0], t[1]) for t in v] for k, v in self._ket_bra_dict.items()}
        return {
            'cirq_type': self.__class__.__name__,
            # JSON requires mappings to have string keys.
            'ket_bra_dict': list(encoded_dict.items()),
        }

    @classmethod
    def _from_json_dict_(cls, ket_bra_dict, **kwargs):
        encoded_dict = dict(ket_bra_dict)
        return cls(ket_bra_dict={k: [tuple(t) for t in v] for k, v in encoded_dict.items()})

    def _value_equality_values_(self) -> Any:
        ket_bra_dict = sorted(self._ket_bra_dict.items())
        encoded_dict = {k: tuple([(t[0], t[1]) for t in v]) for k, v in ket_bra_dict}
        return tuple(encoded_dict.items())
