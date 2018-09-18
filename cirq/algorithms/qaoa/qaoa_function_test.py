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

# imports
import cirq
import numpy as np
from qaoa_function import qaoa_regular_graph

def test_3_regular_swarm():

    solution = qaoa_regular_graph(n_qubits=3, n_connections=2,
                                  p=1,
                                  optimization_method='swarm',
                                  expectation_method='wavefunction')

    assert solution['cost'] > 1.0
