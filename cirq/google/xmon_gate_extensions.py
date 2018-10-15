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

from cirq import ops
from cirq.extension import Extensions
from cirq.google import xmon_gates


xmon_gate_ext = Extensions()

xmon_gate_ext.add_cast(
    desired_type=xmon_gates.XmonGate,
    actual_type=ops.RotXGate,
    conversion=lambda e: xmon_gates.ExpWGate(half_turns=e.half_turns,
                                             axis_half_turns=0))

xmon_gate_ext.add_cast(
    desired_type=xmon_gates.XmonGate,
    actual_type=ops.RotYGate,
    conversion=lambda e: xmon_gates.ExpWGate(half_turns=e.half_turns,
                                             axis_half_turns=0.5))

xmon_gate_ext.add_cast(
    desired_type=xmon_gates.XmonGate,
    actual_type=ops.RotZGate,
    conversion=lambda e: xmon_gates.ExpZGate(half_turns=e.half_turns))
