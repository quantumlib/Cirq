# Copyright 2018 Google LLC
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

from cirq import ops
from cirq.google.xmon_device import XmonDevice
from cirq.google.xmon_qubit import XmonQubit
from cirq.time import Duration

Foxtail = XmonDevice(
    measurement_duration=Duration(nanos=1000),
    exp_w_duration=Duration(nanos=20),
    exp_11_duration=Duration(nanos=50),
    qubits=[XmonQubit(x, y) for x in range(11) for y in range(2)])
