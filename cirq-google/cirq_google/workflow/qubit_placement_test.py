# Copyright 2021 The Cirq Developers
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
import cirq_google as cg

import numpy as np


def test_naive_qubit_placer():
    topo = cirq.TiltedSquareLattice(4, 2)
    qubits = sorted(topo.nodes_to_gridqubits(offset=(5, 3)).values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )

    assert all(q in cg.Sycamore23.qubit_set() for q in circuit.all_qubits())

    qp = cg.NaiveQubitPlacer()
    circuit2, mapping = qp.place_circuit(
        circuit,
        problem_topology=topo,
        shared_rt_info=cg.SharedRuntimeInfo(run_id='1'),
        rs=np.random.RandomState(1),
    )
    assert circuit is not circuit2
    assert circuit == circuit2
    assert all(q in cg.Sycamore23.qubit_set() for q in circuit.all_qubits())
    for k, v in mapping.items():
        assert k == v
