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
import enum


class ResultType(enum.Enum):
    """Expected type of the results from a engine job call.

    Since programs have an embedded Any field, different types
    of data can be passed into a program/job for execution.
    This enum tracks the type of data that was passed in during
    the initial call so that the results can be handled appropriately.

    Program: A single circuit with a single TrialResult.
    Batch: A list of circuits with a list of TrialResults in a BatchResult.
    Calibration: List of CalibrationLayers returning a list of CalibrationResult
    """
    Program = 1
    Batch = 2
    Calibration = 3
