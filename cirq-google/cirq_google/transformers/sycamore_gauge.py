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

"""A gauge transformer for the Sycamore gate."""

import math
from cirq.transformers import GaugeTransformer, GaugeSelector, ConstantGauge
from cirq import ops

from cirq_google.ops.sycamore_gate import SYC

SYCGaugeSelector = GaugeSelector(
    gauges=[
        ConstantGauge(two_qubit_gate=SYC, pre_q0=ops.I, pre_q1=ops.Z, post_q0=ops.Z, post_q1=ops.I),
        ConstantGauge(two_qubit_gate=SYC, pre_q0=ops.Z, pre_q1=ops.I, post_q0=ops.I, post_q1=ops.Z),
        ConstantGauge(two_qubit_gate=SYC, pre_q0=ops.Z, pre_q1=ops.Z, post_q0=ops.Z, post_q1=ops.Z),
        ConstantGauge(
            two_qubit_gate=SYC,
            pre_q0=ops.X,
            pre_q1=ops.X,
            post_q0=(ops.X, ops.rz(-math.pi / 6)),
            post_q1=(ops.X, ops.rz(-math.pi / 6)),
        ),
        ConstantGauge(
            two_qubit_gate=SYC,
            pre_q0=ops.Y,
            pre_q1=ops.Y,
            post_q0=(ops.Y, ops.rz(-math.pi / 6)),
            post_q1=(ops.Y, ops.rz(-math.pi / 6)),
        ),
        ConstantGauge(
            two_qubit_gate=SYC,
            pre_q0=ops.X,
            pre_q1=ops.Y,
            post_q0=(ops.Y, ops.rz(-math.pi / 6)),
            post_q1=(ops.X, ops.rz(-math.pi / 6)),
        ),
        ConstantGauge(
            two_qubit_gate=SYC,
            pre_q0=ops.Y,
            pre_q1=ops.X,
            post_q0=(ops.X, ops.rz(-math.pi / 6)),
            post_q1=(ops.Y, ops.rz(-math.pi / 6)),
        ),
        ConstantGauge(two_qubit_gate=SYC, pre_q0=ops.I, pre_q1=ops.I, post_q0=ops.I, post_q1=ops.I),
    ]
)


SYCGaugeTransformer = GaugeTransformer(target=SYC, gauge_selector=SYCGaugeSelector)
