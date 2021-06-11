import numpy as np
from typing import Any, Dict, Iterable, Optional, Tuple

from cirq import protocols, value
from cirq.ops import raw_types

class MatrixMixture(raw_types.Gate):
    """A generic mixture that can record the index of its selected operator.

    Args:
        mixture: a list of (probability, unitary) pairs
        key: an optional measurement key string for this channel. Simulations
            which select a single unitary to apply will store the index
            of that unitary in the measurement result list with this key.
    """
    def __init__(self, mixture: Iterable[Tuple[float, np.ndarray]], key: Optional[str] = None):
        mixture = list(mixture)
        if not mixture:
            raise ValueError('MatrixMixture must have at least one unitary.')
        if not protocols.approx_eq(sum(p[0] for p in mixture), 1):
            raise ValueError('Unitary probabilities must sum to 1.')
        m0 = mixture[0][1]
        num_qubits = np.log2(m0.size) / 2
        if not num_qubits.is_integer():
            raise ValueError(
                f'Input mixture of shape {m0.shape} does not represent an operator over qubits.'
            )
        self._num_qubits = int(num_qubits)
        for i, op in enumerate(p[1] for p in mixture):
            if not op.size == m0.size:
                raise ValueError(
                    f'Inconsistent unitary sizes: op[0]: {m0.size}, op[{i}]: {op.size}'
                )
        self._mixture = mixture
        self._key = None if key is None else value.MeasurementKey(key)

    @staticmethod
    def from_mixture(mixture: 'protocols.SupportsMixture', key: Optional[str] = None):
        """Creates a copy of a mixture with the given measurement key."""
        return MatrixMixture(mixture=list(protocols.mixture(mixture)), key=key)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MatrixMixture):
            return NotImplemented
        if self._key != other._key:
            return False
        if not np.allclose(
            [m[0] for m in self._mixture],
            [m[0] for m in other._mixture],
        ):
            return False
        return np.allclose(
            [m[1] for m in self._mixture],
            [m[1] for m in other._mixture],
        )

    def num_qubits(self) -> int:
        return self._num_qubits

    def _mixture_(self):
        return self._mixture

    def _measurement_key_(self):
        if self._key is None:
            return NotImplemented
        return self._key

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        if self._key is None:
            return NotImplemented
        if self._key not in key_map:
            return self
        return MatrixMixture(mixture=self._mixture, key=key_map[str(self._key)])

    def _with_key_path_(self, path: Tuple[str, ...]):
        return MatrixMixture(mixture=self._mixture, key=protocols.with_key_path(path))

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['_mixture', '_key'])

    @classmethod
    def _from_json_dict_(cls, _mixture, _key, **kwargs):
        return cls(mixture=_mixture, key=_key)


