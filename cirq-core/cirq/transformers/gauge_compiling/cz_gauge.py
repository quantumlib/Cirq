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

from typing import List

from cirq import circuits, ops
from cirq.ops.common_gates import CZ
from cirq.transformers.gauge_compiling.gauge_compiling import (
    ConstantGauge,
    GaugeSelector,
    GaugeTransformer,
)

CZGaugeSelector = GaugeSelector(
    gauges=[
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.I, pre_q1=ops.I, post_q0=ops.I, post_q1=ops.I),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.I, pre_q1=ops.X, post_q0=ops.Z, post_q1=ops.X),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.I, pre_q1=ops.Y, post_q0=ops.Z, post_q1=ops.Y),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.I, pre_q1=ops.Z, post_q0=ops.I, post_q1=ops.Z),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.X, pre_q1=ops.I, post_q0=ops.X, post_q1=ops.Z),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.X, pre_q1=ops.X, post_q0=ops.Y, post_q1=ops.Y),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.X, pre_q1=ops.Y, post_q0=ops.Y, post_q1=ops.X),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.X, pre_q1=ops.Z, post_q0=ops.X, post_q1=ops.I),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Y, pre_q1=ops.I, post_q0=ops.Y, post_q1=ops.Z),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Y, pre_q1=ops.X, post_q0=ops.X, post_q1=ops.Y),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Y, pre_q1=ops.Y, post_q0=ops.X, post_q1=ops.X),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Y, pre_q1=ops.Z, post_q0=ops.Y, post_q1=ops.I),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Z, pre_q1=ops.I, post_q0=ops.Z, post_q1=ops.I),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Z, pre_q1=ops.X, post_q0=ops.I, post_q1=ops.X),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Z, pre_q1=ops.Y, post_q0=ops.I, post_q1=ops.Y),
        ConstantGauge(two_qubit_gate=CZ, pre_q0=ops.Z, pre_q1=ops.Z, post_q0=ops.Z, post_q1=ops.Z),
    ]
)


def _multi_layer_pull_through_cz(
    left: List[circuits.Moment], moments: List[circuits.Moment]
) -> List[circuits.Moment]:
    # Check all the ops are CZ first
    if not all(op.gate is CZ for moment in moments for op in moment):
        raise ValueError("Input moments must only contains CZ gates.")
    if len(left) > 1:
        raise ValueError("CZ's gauge only has one pre_q0 gate, and one pre_q1 gate.")

    ps: ops.PauliString = ops.PauliString(left[0])
    pulled_through: ops.PauliString = ps.after(moments)
    ret = moments
    ret.append(circuits.Moment([pauli_gate(q) for q, pauli_gate in pulled_through.items()]))
    return ret


CZGaugeTransformer = GaugeTransformer(target=CZ, gauge_selector=CZGaugeSelector)

# Multi-layer pull through version of CZGaugeTransformer
CZGaugeTransformerML = GaugeTransformer(
    target=CZ,
    gauge_selector=CZGaugeSelector,
    multi_layer_pull_thourgh_fn=_multi_layer_pull_through_cz,
)
