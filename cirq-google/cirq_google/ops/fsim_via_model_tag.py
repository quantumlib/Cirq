# Copyright 2024 The Cirq Developers
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
"""A class that can be used to denote FSim gate implementation using polynomial model."""
from typing import Any, Dict

import cirq


class FSimViaModelTag:
    """A tag class to denote FSim gate implementation using polynomial model.

    Without the tag, the translation of FSim gate implementation is possible only for a certain
    angles. For example, when theta=pi/2, phi=0, it translates into the same implementation
    as the SWAP gate. If FSimGate is tagged with this class, the translation will become
    a coupler gate that with proper coupler strength and coupler length via some polynomial
    modelling. Note not all combination of theta and phi in FSim gate are feasible and
    you need the calibration for these angle in advance before using them.
    """

    def __str__(self) -> str:
        return 'FSimViaModelTag()'

    def __repr__(self) -> str:
        return 'cirq_google.FSimViaModelTag()'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, [])

    def __eq__(self, other) -> bool:
        return isinstance(other, FSimViaModelTag)

    def __hash__(self) -> int:
        return hash("FSimViaModelTag")
