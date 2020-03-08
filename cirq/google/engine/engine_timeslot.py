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
import datetime

from typing import Optional
from cirq.google.engine.client.quantum_v1alpha1.gapic import enums

_DEFAULT_TYPE = enums.QuantumTimeSlot.TimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED


@dataclasses.dataclass(frozen=True)
class EngineTimeSlot:
    """A python wrapping of a Quantum Engine timeslot.

    Args:
       start_time: starting datetime of the time slot, usually in local time.
       end_time: ending datetime of the time slot, usually in local time.
       slot_type: type of time slot (reservation, open swim, etc)
       project_id: Google Cloud Platform id of the project, as a string
       maintenance_title: If a MAINTENANCE period, a string title describing the
          type of maintenance being done.
       maintenance_description: If a MAINTENANCE period, a string describing the
          particulars of the maintenancethe title of the slot
    """
    start_time: datetime.datetime
    end_time: datetime.datetime
    slot_type: enums.QuantumTimeSlot.TimeSlotType = _DEFAULT_TYPE
    project_id: Optional[str] = None
    maintenance_title: Optional[str] = None
    maintenance_description: Optional[str] = None
