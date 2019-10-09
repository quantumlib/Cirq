# Copyright 2019 The Cirq Developers
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

import cirq
from cirq.experiments.t1_decay_experiment import t1_decay


class _TimeDependentDecay(cirq.NoiseModel):

    def noisy_moment(self, moment, system_qubits):
        duration = max((op.gate.duration
                        for op in moment.operations
                        if cirq.op_gate_isinstance(op, cirq.WaitGate)),
                       default=cirq.Duration(nanos=1))
        for q in system_qubits:
            yield cirq.amplitude_damp(1 - 0.99**duration.total_nanos()).on(q)
        yield moment


def test_plot_does_not_raise_error():
    results = t1_decay(
        sampler=cirq.DensityMatrixSimulator(noise=_TimeDependentDecay()),
        qubit=cirq.GridQubit(0, 0),
        num_points=3,
        repetitions=10,
        max_delay=cirq.Duration(nanos=100))
    results.plot()
