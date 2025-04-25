# pylint: disable=wrong-or-nonexistent-copyright-notice

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Iterable, Mapping, Tuple, TYPE_CHECKING, Union

import numpy as np

from cirq import linalg, protocols, value
from cirq._compat import proper_repr
from cirq.ops import raw_types

if TYPE_CHECKING:
    import cirq


class MixedUnitaryChannel(raw_types.Gate):
    """A generic mixture that can record the index of its selected operator.

    This type of object is also referred to as a mixed-unitary channel.

    Args:
        mixture: a list of (probability, qubit unitary) pairs
        key: an optional measurement key string for this mixture. Simulations
            which select a single unitary to apply will store the index
            of that unitary in the measurement result list with this key.
        validate: if True, validate that `mixture` describes a valid mixture.
            This validation can be slow; prefer pre-validating if possible.
    """

    def __init__(
        self,
        mixture: Iterable[Tuple[float, np.ndarray]],
        key: Union[str, cirq.MeasurementKey, None] = None,
        validate: bool = False,
    ):
        mixture = list(mixture)
        if not mixture:
            raise ValueError('MixedUnitaryChannel must have at least one unitary.')
        if not protocols.approx_eq(sum(p[0] for p in mixture), 1):
            raise ValueError('Unitary probabilities must sum to 1.')
        m0 = mixture[0][1]
        num_qubits = np.log2(m0.shape[0])
        if not num_qubits.is_integer() or m0.shape[1] != m0.shape[0]:
            raise ValueError(
                f'Input mixture of shape {m0.shape} does not '
                'represent a square operator over qubits.'
            )
        self._num_qubits = int(num_qubits)
        for i, op in enumerate(p[1] for p in mixture):
            if not op.shape == m0.shape:
                raise ValueError(
                    f'Inconsistent unitary shapes: op[0]: {m0.shape}, op[{i}]: {op.shape}'
                )
            if validate and not linalg.is_unitary(op):
                raise ValueError(f'Element {i} of mixture is non-unitary.')
        self._mixture = mixture
        if not isinstance(key, value.MeasurementKey) and key is not None:
            key = value.MeasurementKey(key)
        self._key = key

    @staticmethod
    def from_mixture(
        mixture: protocols.SupportsMixture, key: Union[str, cirq.MeasurementKey, None] = None
    ):
        """Creates a copy of a mixture with the given measurement key."""
        return MixedUnitaryChannel(mixture=list(protocols.mixture(mixture)), key=key)

    def __eq__(self, other) -> bool:
        if not isinstance(other, MixedUnitaryChannel):
            return NotImplemented
        if self._key != other._key:
            return False
        if not np.allclose([m[0] for m in self._mixture], [m[0] for m in other._mixture]):
            return False
        return np.allclose(
            np.asarray([m[1] for m in self._mixture]), np.asarray([m[1] for m in other._mixture])
        )

    def num_qubits(self) -> int:
        return self._num_qubits

    def _mixture_(self):
        return self._mixture

    def _measurement_key_name_(self) -> str:
        if self._key is None:
            return NotImplemented
        return str(self._key)

    def _measurement_key_obj_(self) -> cirq.MeasurementKey:
        if self._key is None:
            return NotImplemented
        return self._key

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        if self._key is None:
            return NotImplemented
        if self._key not in key_map:
            return self
        return MixedUnitaryChannel(mixture=self._mixture, key=key_map[str(self._key)])

    def _with_key_path_(self, path: Tuple[str, ...]):
        return MixedUnitaryChannel(
            mixture=self._mixture, key=protocols.with_key_path(self._key, path)
        )

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return MixedUnitaryChannel(
            mixture=self._mixture, key=protocols.with_key_path_prefix(self._key, prefix)
        )

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet[cirq.MeasurementKey]
    ):
        return MixedUnitaryChannel(
            mixture=self._mixture, key=protocols.with_rescoped_keys(self._key, path, bindable_keys)
        )

    def __str__(self):
        if self._key is not None:
            return f'MixedUnitaryChannel({self._mixture}, key={self._key})'
        return f'MixedUnitaryChannel({self._mixture})'

    def __repr__(self):
        unitary_tuples = [
            '(' + repr(op[0]) + ', ' + proper_repr(op[1]) + ')' for op in self._mixture
        ]
        args = [f'mixture=[{", ".join(unitary_tuples)}]']
        if self._key is not None:
            args.append(f'key=\'{self._key}\'')
        return f'cirq.MixedUnitaryChannel({", ".join(args)})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['_mixture', '_key'])

    @classmethod
    def _from_json_dict_(cls, _mixture, _key, **kwargs):
        mix_pairs = [(m[0], np.asarray(m[1])) for m in _mixture]
        return cls(mixture=mix_pairs, key=_key)
