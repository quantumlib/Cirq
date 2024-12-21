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

"""A Gauge Transformer for the CZ gate."""

from cirq.transformers.gauge_compiling.gauge_compiling import (
    GaugeTransformer,
    GaugeSelector,
    ConstantGauge,
)
from cirq.ops.common_gates import CZ
from cirq import ops

CZGaugeSelector = GaugeSelector(
    gauges=[
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.I,
            pre_q1=ops.I,
            post_q0=ops.I,
            post_q1=ops.I,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.I,
            pre_q1=ops.X,
            post_q0=ops.Z,
            post_q1=ops.X,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.I,
            pre_q1=ops.Y,
            post_q0=ops.Z,
            post_q1=ops.Y,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.I,
            pre_q1=ops.Z,
            post_q0=ops.I,
            post_q1=ops.Z,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.X,
            pre_q1=ops.I,
            post_q0=ops.X,
            post_q1=ops.Z,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.X,
            pre_q1=ops.X,
            post_q0=ops.Y,
            post_q1=ops.Y,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.X,
            pre_q1=ops.Y,
            post_q0=ops.Y,
            post_q1=ops.X,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.X,
            pre_q1=ops.Z,
            post_q0=ops.X,
            post_q1=ops.I,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Y,
            pre_q1=ops.I,
            post_q0=ops.Y,
            post_q1=ops.Z,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Y,
            pre_q1=ops.X,
            post_q0=ops.X,
            post_q1=ops.Y,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Y,
            pre_q1=ops.Y,
            post_q0=ops.X,
            post_q1=ops.X,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Y,
            pre_q1=ops.Z,
            post_q0=ops.Y,
            post_q1=ops.I,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Z,
            pre_q1=ops.I,
            post_q0=ops.Z,
            post_q1=ops.I,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Z,
            pre_q1=ops.X,
            post_q0=ops.I,
            post_q1=ops.X,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Z,
            pre_q1=ops.Y,
            post_q0=ops.I,
            post_q1=ops.Y,
            support_sweep=True,
        ),
        ConstantGauge(
            two_qubit_gate=CZ,
            pre_q0=ops.Z,
            pre_q1=ops.Z,
            post_q0=ops.Z,
            post_q1=ops.Z,
            support_sweep=True,
        ),
    ]
)

CZGaugeTransformer = GaugeTransformer(target=CZ, gauge_selector=CZGaugeSelector)
