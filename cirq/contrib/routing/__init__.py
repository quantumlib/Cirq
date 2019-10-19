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
"""Utilities for routing circuits on devices"""

from cirq.contrib.routing.device import (
    xmon_device_to_graph,
    get_linear_device_graph,
    get_grid_device_graph,
    gridqubits_to_graph_device,
    nx_qubit_layout,
)
from cirq.contrib.routing.router import (route_circuit, ROUTERS)
from cirq.contrib.routing.swap_network import SwapNetwork
from cirq.contrib.routing.utils import (
    get_circuit_connectivity,
    is_valid_routing,
    ops_are_consistent_with_device_graph,
)
