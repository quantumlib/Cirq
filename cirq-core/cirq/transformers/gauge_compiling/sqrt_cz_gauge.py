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

"""A Gauge transformer for CZ**0.5 gate."""

from cirq.transformers.gauge_compiling.gauge_compiling import (
    GaugeTransformer,
    GaugeSelector,
    ConstantGauge,
)
from cirq.ops.common_gates import CZ
from cirq import ops

SqrtCZGaugeSelector = GaugeSelector(
    gauges=[ConstantGauge(pre_q0=ops.X, post_q0=ops.X, post_q1=ops.Z**0.5, two_qubit_gate=CZ**-0.5)]
)

SqrtCZGaugeTransformer = GaugeTransformer(target=CZ**0.5, gauge_selector=SqrtCZGaugeSelector)
