# Copyright 2018 The Cirq Developers
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

from unittest import mock
import cirq
import cirq.google as cg
from cirq.google import line_on_device
from cirq.google.line.placement import GridQubitLineTuple


def test_anneal_method_calls_anneal_search():
    q00 = cirq.GridQubit(0, 0)
    q01 = cirq.GridQubit(0, 1)
    q03 = cirq.GridQubit(0, 3)
    device = cg.XmonDevice(
        cirq.Duration(), cirq.Duration(), cirq.Duration(), qubits=[q00, q01, q03]
    )
    length = 2
    method = cg.AnnealSequenceSearchStrategy()

    with mock.patch.object(method, 'place_line') as place_line:
        sequences = GridQubitLineTuple((q00, q01))
        place_line.return_value = sequences

        assert line_on_device(device, length, method) == sequences
        place_line.assert_called_once_with(device, length)
