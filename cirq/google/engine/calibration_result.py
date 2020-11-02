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
#
import dataclasses
import datetime
from typing import TYPE_CHECKING

import cirq.google.api.v2.calibration_pb2 as calibration_pb2

if TYPE_CHECKING:
    import cirq


@dataclasses.dataclass
class CalibrationResult:
    """Python implementation of the proto found in
    cirq.google.api.v2.calibration_pb2.CalibrationLayerResult for use
    in Engine calls."""
    code: calibration_pb2.CalibrationLayerCode
    error_message: str
    token: str
    valid_until: datetime.datetime
    metrics: 'cirq.google.Calibration'
