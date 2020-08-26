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
import cirq.contrib.routing as ccr
from cirq.contrib.routing.multi_prog_mapping import *


def test_bad_args():
    circuit = cirq.testing.random_circuit(4, 2, 0.5, random_state=5)
    device_graph = ccr.get_grid_device_graph(3, 2)
    
    multi_prog_map(circuit, device_graph)

    