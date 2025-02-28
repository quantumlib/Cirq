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

"""Tools for representing a device as an edge-labelled graph."""

from cirq.contrib.graph_device.hypergraph import UndirectedHypergraph as UndirectedHypergraph

from cirq.contrib.graph_device.graph_device import (
    is_undirected_device_graph as is_undirected_device_graph,
    is_crosstalk_graph as is_crosstalk_graph,
    FixedDurationUndirectedGraphDeviceEdge as FixedDurationUndirectedGraphDeviceEdge,
    UndirectedGraphDevice as UndirectedGraphDevice,
    UnconstrainedUndirectedGraphDeviceEdge as UnconstrainedUndirectedGraphDeviceEdge,
)

from cirq.contrib.graph_device.uniform_graph_device import (
    uniform_undirected_graph_device as uniform_undirected_graph_device,
    uniform_undirected_linear_device as uniform_undirected_linear_device,
)
