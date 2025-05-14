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

import numpy as np

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


def _multi_moment_pull_through(
    moments: List[circuits.Moment], rng: np.random.Generator
) -> List[circuits.Moment]:
    # Check all the ops are CZ first
    if not all(op.gate == CZ for moment in moments for op in moment):
        raise ValueError(f"Input moments must only contain CZ gates:\nmoments = {moments}.")

    left: List[ops.Operation] = [
        rng.choice([ops.I, ops.X, ops.Y, ops.Z]).on(q)
        for q in circuits.Circuit(moments).all_qubits()
    ]
    if not left:
        return moments

    ps: ops.PauliString = ops.PauliString(left)
    pulled_through: ops.PauliString = ps.after(moments)
    ret = [circuits.Moment(left)] + moments
    ret.append(circuits.Moment([pauli_gate(q) for q, pauli_gate in pulled_through.items()]))
    return ret


CZGaugeTransformer = GaugeTransformer(target=CZ, gauge_selector=CZGaugeSelector)

# Multi-moments pull through version of CZGaugeTransformer
CZGaugeTransformerMM = GaugeTransformer(
    target=CZ,
    gauge_selector=CZGaugeSelector,
    multi_moment_pull_thourgh_fn=_multi_moment_pull_through,
)
