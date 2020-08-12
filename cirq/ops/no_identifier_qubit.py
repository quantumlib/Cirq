# Copyright 2018 The Cirq Developers
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
from typing import Any, Dict

from cirq import protocols
from cirq.ops import raw_types


class NoIdentifierQubit(raw_types.Qid):
    """A qubit with no identifier - created to test QubitAsQid
    """

    def __init__(self) -> None:
        return None

    def _comparison_key(self):
        return 0

    @property
    def dimension(self) -> int:
        return 2

    def __repr__(self) -> str:
        return f'cirq.NoIdentifierQubit()'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, [])
