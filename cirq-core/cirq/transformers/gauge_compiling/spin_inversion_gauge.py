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

"""The spin inversion gauge transformer."""

from cirq.transformers.gauge_compiling.gauge_compiling import (
    GaugeTransformer,
    GaugeSelector,
    SameGateGauge,
)
from cirq import ops

SpinInversionGaugeSelector = GaugeSelector(
    gauges=[
        SameGateGauge(pre_q0=ops.X, post_q0=ops.X, pre_q1=ops.X, post_q1=ops.X),
        SameGateGauge(),
    ]
)

SpinInversionGaugeTransformer = GaugeTransformer(
    target=ops.GateFamily(ops.ZZPowGate), gauge_selector=SpinInversionGaugeSelector
)
