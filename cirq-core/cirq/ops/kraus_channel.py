# pylint: disable=wrong-or-nonexistent-copyright-notice
from typing import Any, Dict, FrozenSet, Iterable, Tuple, TYPE_CHECKING, Union
import numpy as np

from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


# TODO(#3241): support qudits and non-square operators.
class KrausChannel(raw_types.Gate):
    """A generic channel that can record the index of its selected operator.

    Args:
        kraus_ops: a list of Kraus operators, formatted as numpy array.
            Currently, only square-matrix operators on qubits (not qudits) are
            supported by this type.
        key: an optional measurement key string for this channel. Simulations
            which select a single Kraus operator to apply will store the index
            of that operator in the measurement result list with this key.
        validate: if True, validate that `kraus_ops` describe a valid channel.
            This validation can be slow; prefer pre-validating if possible.
    """

    def __init__(
        self,
        kraus_ops: Iterable[np.ndarray],
        key: Union[str, 'cirq.MeasurementKey', None] = None,
        validate: bool = False,
    ):
        kraus_ops = list(kraus_ops)
        if not kraus_ops:
            raise ValueError('KrausChannel must have at least one operation.')
        num_qubits = np.log2(kraus_ops[0].shape[0])
        if not num_qubits.is_integer() or kraus_ops[0].shape[1] != kraus_ops[0].shape[0]:
            raise ValueError(
                f'Input Kraus ops of shape {kraus_ops[0].shape} does not '
                'represent a square operator over qubits.'
            )
        self._num_qubits = int(num_qubits)
        for i, op in enumerate(kraus_ops):
            if not op.shape == kraus_ops[0].shape:
                raise ValueError(
                    'Inconsistent Kraus operator shapes: '
                    f'op[0]: {kraus_ops[0].shape}, op[{i}]: {op.shape}'
                )
        if validate and not linalg.is_cptp(kraus_ops=kraus_ops):
            raise ValueError('Kraus operators do not describe a CPTP map.')
        self._kraus_ops = kraus_ops
        if not isinstance(key, value.MeasurementKey) and key is not None:
            key = value.MeasurementKey(key)
        self._key = key

    @staticmethod
    def from_channel(channel: 'KrausChannel', key: Union[str, 'cirq.MeasurementKey', None] = None):
        """Creates a copy of a channel with the given measurement key."""
        return KrausChannel(kraus_ops=list(protocols.kraus(channel)), key=key)

    def __eq__(self, other) -> bool:
        # TODO(#3241): provide a protocol to test equivalence between channels,
        # ignoring measurement keys and channel/mixture distinction
        if not isinstance(other, KrausChannel):
            return NotImplemented
        if self._key != other._key:
            return False
        return np.allclose(self._kraus_ops, other._kraus_ops)

    def num_qubits(self) -> int:
        return self._num_qubits

    def _kraus_(self):
        return self._kraus_ops

    def _measurement_key_name_(self) -> str:
        if self._key is None:
            return NotImplemented
        return str(self._key)

    def _measurement_key_obj_(self) -> 'cirq.MeasurementKey':
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

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return KrausChannel(
            kraus_ops=self._kraus_ops, key=protocols.with_key_path_prefix(self._key, prefix)
        )

    def _with_rescoped_keys_(
        self,
        path: Tuple[str, ...],
        bindable_keys: FrozenSet['cirq.MeasurementKey'],
    ):
        return KrausChannel(
            kraus_ops=self._kraus_ops,
            key=protocols.with_rescoped_keys(self._key, path, bindable_keys),
        )

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
