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


def neutral_atom_gateset(max_parallel_z=None, max_parallel_xy=None):
    return ops.Gateset(
        ops.AnyIntegerPowerGateFamily(ops.CNotPowGate),
        ops.AnyIntegerPowerGateFamily(ops.CCNotPowGate),
        ops.AnyIntegerPowerGateFamily(ops.CZPowGate),
        ops.AnyIntegerPowerGateFamily(ops.CCZPowGate),
        ops.ParallelGateFamily(ops.ZPowGate, max_parallel_allowed=max_parallel_z),
        ops.ParallelGateFamily(ops.XPowGate, max_parallel_allowed=max_parallel_xy),
        ops.ParallelGateFamily(ops.YPowGate, max_parallel_allowed=max_parallel_xy),
        ops.ParallelGateFamily(ops.PhasedXPowGate, max_parallel_allowed=max_parallel_xy),
        ops.MeasurementGate,
        ops.IdentityGate,
        unroll_circuit_op=False,
    )
