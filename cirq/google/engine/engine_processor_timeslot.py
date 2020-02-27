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
import enum

from typing import Optional


class EngineProcessorTimeSlotType(enum.Enum):
    TIME_SLOT_TYPE_UNSPECIFIED = 0

    # Time reserved for processor admins to run calibration or make changes.
    MAINTENANCE = 1

    # Time for anyone to create jobs.
    OPEN_SWIM = 2

    # Reservation is time when one project has exclusive hold to create jobs.
    RESERVATION = 3

    # Available for reservations to be made
    UNALLOCATED = 4


_DEFAULT_SLOT_TYPE = EngineProcessorTimeSlotType.TIME_SLOT_TYPE_UNSPECIFIED

@dataclasses.dataclass
class EngineProcessorTimeSlot:
    """A python wrapping of a Quantum Engine timeslot.
  """
    start_seconds: int
    end_seconds: int
    slot_type: EngineProcessorTimeSlotType = _DEFAULT_SLOT_TYPE
    project_id: Optional[int] = None
    maintenance_title: Optional[str] = None
    maintenance_description: Optional[str] = None
