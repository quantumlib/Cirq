# Copyright 2019 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cirq
from cirq.interop.quirk.cells.testing import assert_url_to_circuit_returns


def test_measurement_gates():
    a, b, c = cirq.LineQubit.range(3)
    assert_url_to_circuit_returns(
        '{"cols":[["Measure","Measure"],["Measure","Measure"]]}',
        cirq.Circuit(
            cirq.measure(a, key='row=0,col=0'),
            cirq.measure(b, key='row=1,col=0'),
            cirq.measure(a, key='row=0,col=1'),
            cirq.measure(b, key='row=1,col=1'),
        ))
    assert_url_to_circuit_returns(
        '{"cols":[["XDetector","YDetector","ZDetector"]]}',
        cirq.Circuit(
            cirq.X(b)**-0.5,
            cirq.Y(a)**0.5,
            cirq.Moment([
                cirq.measure(a, key='row=0,col=0'),
                cirq.measure(b, key='row=1,col=0'),
                cirq.measure(c, key='row=2,col=0'),
            ]),
            cirq.Y(a)**-0.5,
            cirq.X(b)**0.5,
        ))
