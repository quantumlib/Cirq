# Copyright 2020 The Cirq Developers
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
import dataclasses

from typing import Optional
from cirq.google.engine.client.quantum_v1alpha1.gapic import enums

_DEFAULT_TYPE = enums.QuantumProcessor.TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED


@dataclasses.dataclass(frozen=True)
class EngineTimeSlot:
    """A python wrapping of a Quantum Engine timeslot.
  """
    start_seconds: int
    end_seconds: int
    slot_type: enums.QuantumProcessor.TimeSlotType = _DEFAULT_TYPE
    project_id: Optional[int] = None
    maintenance_title: Optional[str] = None
    maintenance_description: Optional[str] = None
