from typing import Any, Dict, Iterable, Optional, Tuple
import numpy as np

from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types


class KrausChannel(raw_types.Gate):
    """A generic channel that can record the index of its selected operator.

    Args:
        kraus_ops: a list of Kraus operators, formatted as numpy array.
        key: an optional measurement key string for this channel. Simulations
            which select a single Kraus operator to apply will store the index
            of that operator in the measurement result list with this key.
    """

    def __init__(self, kraus_ops: Iterable[np.ndarray], key: Optional[str] = None):
        # TODO: validate channel representations (issue #2271)
        kraus_ops = list(kraus_ops)
        if not kraus_ops:
            raise ValueError('KrausChannel must have at least one operation.')
        num_qubits = np.log2(kraus_ops[0].size) / 2
        if not num_qubits.is_integer():
            raise ValueError(
                f'Input Kraus ops of shape {kraus_ops[0].shape} does not '
                'represent an operator over qubits.'
            )
        self._num_qubits = int(num_qubits)
        for i, op in enumerate(kraus_ops):
            if not op.size == kraus_ops[0].size:
                raise ValueError(
                    'Inconsistent Kraus operator sizes: '
                    f'op[0]: {kraus_ops[0].size}, op[{i}]: {op.size}'
                )
        self._kraus_ops = kraus_ops
        if key is None:
            self._key = None
        elif isinstance(key, value.MeasurementKey):
            self._key = key
        else:
            self._key = value.MeasurementKey(key)

    @staticmethod
    def from_channel(channel: 'protocols.SupportsChannel', key: Optional[str] = None):
        """Creates a copy of a channel with the given measurement key."""
        return KrausChannel(kraus_ops=list(protocols.kraus(channel)), key=key)

    def __eq__(self, other) -> bool:
        if not isinstance(other, KrausChannel):
            return NotImplemented
        if self._key != other._key:
            return False
        return np.allclose(self._kraus_ops, other._kraus_ops)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _kraus_(self):
        return self._kraus_ops

    def _measurement_key_(self):
        if self._key is None:
            return NotImplemented
        return self._key

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        if self._key is None:
            return NotImplemented
        if self._key not in key_map:
            return self
        return KrausChannel(kraus_ops=self._kraus_ops, key=key_map[str(self._key)])

    def _with_key_path_(self, path: Tuple[str, ...]):
        return KrausChannel(kraus_ops=self._kraus_ops, key=protocols.with_key_path(self._key, path))

    def __str__(self):
        if self._key is not None:
            return f'KrausChannel({self._kraus_ops}, key={self._key})'
        return f'KrausChannel({self._kraus_ops})'

    def __repr__(self):
        args = ['kraus_ops=[' + ', '.join(proper_repr(op) for op in self._kraus_ops) + ']']
        if self._key is not None:
            args.append(f'key=\'{self._key}\'')
        return f'cirq.KrausChannel({", ".join(args)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['_kraus_ops', '_key'])

    @classmethod
    def _from_json_dict_(cls, _kraus_ops, _key, **kwargs):
        ops = [np.asarray(op) for op in _kraus_ops]
        return cls(kraus_ops=ops, key=_key)
