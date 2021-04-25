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

from typing import Optional

import dataclasses

MEASUREMENT_KEY_SEPARATOR = ':'


@dataclasses.dataclass(frozen=True)
class MeasurementKey:
    """A class representing a Measurement Key.

    Wraps a string key and provides validation. If you just want the string measurement key, simply
    call `str()` on this.
    TODO: Change allow_nested_key to is_nested_key when the nested key callers start interacting
        with `MeasurementKey` directly. That way we can have tighter validation and won't need to
        store allow_nested_key, is_nested_key can just be derived from name itself. This will
        happen in Phase 2a detailed at
        https://tinyurl.com/structured-measurement-keys#heading=h.zafcj653k11m

    Args:
        name: The string representation of the key.
        allow_nested_key: Whether nesting-defining separators are allowed in the key name.
    """

    _hash: Optional[int] = dataclasses.field(default=None, init=False)

    name: str
    allow_nested_key: bool = False

    def __post_init__(self):
        if not self.allow_nested_key and MEASUREMENT_KEY_SEPARATOR in self.name:
            raise ValueError(
                f'Measurement key {self.name} invalid. "{MEASUREMENT_KEY_SEPARATOR}" is not '
                'allowed in measurement key constructor. If you want to use this character to '
                'specify nested measurement keys, set the allow_nested_key option.'
            )

    def __eq__(self, other) -> bool:
        if isinstance(other, (MeasurementKey, str)):
            return str(self) == str(other)
        return NotImplemented

    def __repr__(self):
        return f'cirq.MeasurementKey(name={self.name})'

    def __str__(self):
        return self.name

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(self, '_hash', hash(self.name))
        return self._hash

    def _json_dict_(self):
        return {
            'cirq_type': 'MeasurementKey',
            'name': self.name,
            # allow_nested_key needs to be serialized to allow bypassing separator validation when
            # de-serializing.
            'allow_nested_key': self.allow_nested_key,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        name,
        allow_nested_key,
        **kwargs,
    ):
        return cls(name, allow_nested_key)
