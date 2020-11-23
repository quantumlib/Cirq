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
#
import dataclasses
from typing import Dict, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class CalibrationLayer:
    """Python implementation of the proto found in
    cirq.google.api.v2.calibration_pb2.FocusedCalibrationLayer for use
    in Engine calls."""
    calibration_type: str
    program: 'cirq.Circuit'
    args: Dict[str, Union[str, float]]
