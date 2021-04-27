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

    Wraps a string key. If you just want the string measurement key, simply call `str()` on this.

    Args:
        name: The string representation of the key.
    """

    _hash: Optional[int] = dataclasses.field(default=None, init=False)

    name: str

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
        }

    @classmethod
    def _from_json_dict_(
        cls,
        name,
        **kwargs,
    ):
        return cls(name)
