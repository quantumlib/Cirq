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


def test_displays():
    assert_url_to_circuit_returns(
        '{"cols":[["Amps2"],[1,"Amps3"],["Chance"],'
        '["Chance2"],["Density"],["Density3"],'
        '["Sample4"],["Bloch"],["Sample2"]'
        ']}',
        cirq.Circuit(),
    )
