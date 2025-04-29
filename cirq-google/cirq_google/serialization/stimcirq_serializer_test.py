# Copyright 2025 The Cirq Developers
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
import pytest

import cirq
from cirq_google.serialization.stimcirq_serializer import StimCirqSerializer


def test_stimcirq_gates():
    stimcirq = pytest.importorskip("stimcirq")
    serializer = StimCirqSerializer()
    q = cirq.q(1, 2)
    q2 = cirq.q(2, 2)
    # Initialize stimcirq operations after importerskip
    ops = [
        stimcirq.CumulativeObservableAnnotation(parity_keys=["m"], observable_index=123),
        stimcirq.MeasureAndOrResetGate(
            measure=True,
            reset=False,
            basis='Z',
            invert_measure=True,
            key='mmm',
            measure_flip_probability=0.125,
        )(q2),
        stimcirq.ShiftCoordsAnnotation([1.0, 2.0]),
        stimcirq.SweepPauli(stim_sweep_bit_index=2, cirq_sweep_symbol='t', pauli=cirq.X)(q),
        stimcirq.SweepPauli(stim_sweep_bit_index=3, cirq_sweep_symbol='y', pauli=cirq.Y)(q),
        stimcirq.SweepPauli(stim_sweep_bit_index=4, cirq_sweep_symbol='t', pauli=cirq.Z)(q),
        stimcirq.TwoQubitAsymmetricDepolarizingChannel([0.05] * 15)(q, q2),
        stimcirq.CZSwapGate()(q, q2),
        stimcirq.CXSwapGate(inverted=True)(q, q2),
        stimcirq.DetAnnotation(parity_keys=["m"]),
    ]
    for op in ops:
        assert serializer.can_serialize_operation(op)
        proto = serializer.to_proto(op, constants=[], raw_constants={})
        assert proto.internalgate.module == 'stimcirq'
        assert proto.internalgate.name in [type(op).__name__, type(op.gate).__name__]
