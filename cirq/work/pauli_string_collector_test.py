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


def test_pauli_string_sample_collector():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliStringCollector(cirq.Circuit.from_ops(
        cirq.H(a), cirq.CNOT(a, b), cirq.X(a), cirq.Z(b)),
                                        samples_per_term=100,
                                        terms=cirq.LinearDict({
                                            cirq.X(a) * cirq.X(b):
                                            1,
                                            cirq.Y(a) * cirq.Y(b):
                                            -16,
                                            cirq.Z(a) * cirq.Z(b):
                                            4,
                                        }))
    completion = p.collect_async(sampler=cirq.Simulator())
    cirq.testing.assert_asyncio_will_have_result(completion, None)
    assert p.estimated_energy() == 11
