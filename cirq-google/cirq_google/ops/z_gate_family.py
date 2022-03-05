# Copyright 2022 The Cirq Developers
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

from typing import Union

import cirq
from cirq.ops import raw_types
from cirq_google.ops.physical_z_tag import PhysicalZTag


_PHYSICAL_Z_TAG = PhysicalZTag()


class ZGateFamily(cirq.GateFamily):
    """A GateFamily which accepts either virtual or physical Z gates from Google devices.

    If `physical_z` is true, accepts physical Z gates, i.e. `TaggedOperations` containing a Z gate
    and a `cirq_google.PhysicalZTag`.
    Otherwise, Accepts virtual Z gates, i.e. Z gates without a `cirq_google.PhysicalZTag`.
    """

    def __init__(self, physical_z=False) -> None:
        self._physical_z = physical_z
        super().__init__(cirq.Z, ignore_global_phase=True)

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if (
            isinstance(item, cirq.TaggedOperation)
            and item.gate is not None
            and super().__contains__(item.gate)
        ):
            return self._physical_z == (_PHYSICAL_Z_TAG in item.tags)

        if not self._physical_z:
            return super().__contains__(item)

        return False

    def _default_name(self) -> str:
        return f"ZGateFamily({'Physical' if self._physical_z else 'Virtual'})"

    def _default_description(self) -> str:
        if self._physical_z:
            return 'Accepts TaggedOperations containing a Z gate and a cirq_google.PhysicalZTag'
        return 'Accepts a virtual Z gate, i.e. a Z gate without cirq_google.PhysicalZTag'

    def __repr__(self) -> str:
        return 'cirq_google.ZGateFamily(' f'physical_z={self._physical_z})'

    def _json_dict_(self):
        return {
            'physical_z': self._physical_z,
        }

    @classmethod
    def _from_json_dict_(cls, physical_z, **kwargs):
        return cls(physical_z)
