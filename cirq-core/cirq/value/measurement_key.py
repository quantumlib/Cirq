# Copyright 2021 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, Iterable, Optional, Tuple, TYPE_CHECKING

import dataclasses

if TYPE_CHECKING:
    import cirq

MEASUREMENT_KEY_SEPARATOR = ':'
QUBIT_SEPARATOR = ','


def default_measurement_key_str(qubits: Iterable['cirq.Qid']) -> str:
    return QUBIT_SEPARATOR.join(str(q) for q in qubits)


@dataclasses.dataclass(frozen=True)
class MeasurementKey:
    """A class representing a Measurement Key.

    Wraps a string key with additional metadata. If you just want the string measurement key,
    simply call `str()` on this.

    Args:
        name: The string representation of the key.
        path: The path to this key in a circuit. In a multi-level circuit (one with repeated or
            nested subcircuits), we need to differentiate the keys that occur multiple times. The
            path is used to create such fully qualified unique measurement key based on where it
            occurs in the circuit. The path is outside-to-in, the outermost subcircuit identifier
            appears first in the tuple.
        qubits: Tuple of qubits the key operates on. Used to create a default base key string iff
            the name is empty. Also supports qubit remapping, which changes the actual string key,
            if needed. If a `name` is explicitly specified, this does not have any effect on the
            object but is still used during comparisons.
    """

    _hash: Optional[int] = dataclasses.field(default=None, init=False)
    _str: Optional[str] = dataclasses.field(default=None, init=False)

    name: str = dataclasses.field(default='')
    path: Tuple[str, ...] = dataclasses.field(default_factory=tuple)
    qubits: Tuple['cirq.Qid', ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        if MEASUREMENT_KEY_SEPARATOR in self.name:
            raise ValueError(
                f'Invalid key name: {self.name}\n{MEASUREMENT_KEY_SEPARATOR} is not allowed in '
                'MeasurementKey. If this is a nested key string, use '
                '`MeasurementKey.parse_serialized` for correct behavior.'
            )

    def replace(self, **changes) -> 'MeasurementKey':
        """Returns a copy of this MeasurementKey with the specified changes."""
        return dataclasses.replace(self, **changes)

    def is_qubit_based_key(self) -> bool:
        """Returns True if the current base key string will be determined based on the acted upon
        qubits. Equivalent to whether the `name` field is empty."""
        return self.name == ''

    def base_key_str(self) -> str:
        """The base key string for this `MeasurementKey` i.e. the key without the parent path."""
        return default_measurement_key_str(self.qubits) if self.is_qubit_based_key() else self.name

    def __eq__(self, other) -> bool:
        if isinstance(other, str):
            return str(self) == other
        if isinstance(other, MeasurementKey):
            return (
                self.name == other.name and self.path == other.path and self.qubits == other.qubits
            )
        return NotImplemented

    def __repr__(self):
        args = []
        if self.path:
            args.append(f'path={self.path!r}')
        if self.name:
            args.append(f'name={self.name!r}')
        if self.qubits:
            args.append(f'qubits={tuple(repr(q) for q in self.qubits)}')
        arg_list = ', '.join(args)
        return f'cirq.MeasurementKey({arg_list})'

    def __str__(self):
        if self._str is None:
            object.__setattr__(
                self, '_str', MEASUREMENT_KEY_SEPARATOR.join(self.path + (self.base_key_str(),))
            )
        return self._str

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(self, '_hash', hash(str(self)))
        return self._hash

    def _json_dict_(self):
        return {
            'cirq_type': 'MeasurementKey',
            'name': self.name,
            'path': self.path,
            'qubits': self.qubits,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        name,
        path,
        qubits,
        **kwargs,
    ):
        return cls(name=name, path=tuple(path), qubits=tuple(qubits))

    @classmethod
    def parse_serialized(cls, key_str: str):
        """Parses the serialized string representation of `Measurementkey` into a `MeasurementKey`.

        This is the only way to construct a `MeasurementKey` from a nested string representation
        (where the path is joined to the key name by the `MEASUREMENT_KEY_SEPARATOR`)"""
        components = key_str.split(MEASUREMENT_KEY_SEPARATOR)
        # Even a qubit-based serialized key is parsed as a regular name-based key since the qubit
        # serialization is lossful and creating qubits from those strings is not possible. We could
        # have a lossless serialization to fix this but it would change the default constructed
        # keys string behavior and might break client code that relies on the current default key
        # string structure. That said, the only limitation of not parsing qubit keys would be that
        # qubit remapping will not remap the key.
        return MeasurementKey(name=components[-1], path=tuple(components[:-1]))

    def _with_key_path_(self, path: Tuple[str, ...]):
        if path == self.path:
            return self
        return self.replace(path=path)

    def with_key_path_prefix(self, path_component: str):
        """Adds the input path component to the start of the path.

        Useful when constructing the path from inside to out (in case of nested subcircuits),
        recursively.
        """
        return self._with_key_path_((path_component,) + self.path)

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]):
        base_key = self.base_key_str()
        if base_key not in key_map:
            return self
        remapped_key_name = key_map[base_key]
        if not remapped_key_name:
            raise ValueError(
                f'Invalid remapping from key {base_key} to empty string. '
                'Remapped key should be non-empty.'
            )
        return self.replace(name=remapped_key_name)

    def with_qubits(self, qubits: Tuple['cirq.Qid', ...]):
        """Updates the MeasurementKey to operate on the input qubits."""
        return self.replace(qubits=qubits)
