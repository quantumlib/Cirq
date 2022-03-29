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
import pytest

import cirq
import cirq_google as cg

import numpy as np


class FakeDevice(cirq.Device):
    def __init__(self):
        self.qubits = cirq.GridQubit.rect(2, 8)
        neighbors = [(a, b) for a in self.qubits for b in self.qubits if a.is_adjacent(b)]
        self._metadata = cirq.GridDeviceMetadata(neighbors, cirq.Gateset(cirq.H))

    @property
    def metadata(self):
        return self._metadata


def test_naive_qubit_placer():
    topo = cirq.TiltedSquareLattice(4, 2)
    qubits = sorted(topo.nodes_to_gridqubits(offset=(5, 3)).values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )

    assert all(q in cg.Sycamore23.metadata.qubit_set for q in circuit.all_qubits())

    qp = cg.NaiveQubitPlacer()
    circuit2, mapping = qp.place_circuit(
        circuit,
        problem_topology=topo,
        shared_rt_info=cg.SharedRuntimeInfo(run_id='1'),
        rs=np.random.RandomState(1),
    )
    assert circuit is not circuit2
    assert circuit == circuit2
    assert all(q in cg.Sycamore23.metadata.qubit_set for q in circuit2.all_qubits())
    for k, v in mapping.items():
        assert k == v


def test_random_device_placer_tilted_square_lattice():
    topo = cirq.TiltedSquareLattice(4, 2)
    qubits = sorted(topo.nodes_to_gridqubits().values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )
    assert not all(q in cg.Sycamore23.metadata.qubit_set for q in circuit.all_qubits())

    qp = cg.RandomDevicePlacer()
    circuit2, mapping = qp.place_circuit(
        circuit,
        problem_topology=topo,
        shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=cg.Sycamore23),
        rs=np.random.RandomState(1),
    )
    assert circuit is not circuit2
    assert circuit != circuit2
    assert all(q in cg.Sycamore23.metadata.qubit_set for q in circuit2.all_qubits())
    for k, v in mapping.items():
        assert k != v


def test_random_device_placer_line():
    topo = cirq.LineTopology(8)
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.testing.random_circuit(qubits, n_moments=8, op_density=1.0, random_state=52)

    qp = cg.RandomDevicePlacer()
    circuit2, mapping = qp.place_circuit(
        circuit,
        problem_topology=topo,
        shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=cg.Sycamore23),
        rs=np.random.RandomState(1),
    )
    assert circuit is not circuit2
    assert circuit != circuit2
    assert all(q in cg.Sycamore23.metadata.qubit_set for q in circuit2.all_qubits())
    for k, v in mapping.items():
        assert k != v


def test_random_device_placer_repr():
    cirq.testing.assert_equivalent_repr(cg.RandomDevicePlacer(), global_vals={'cirq_google': cg})


def test_random_device_placer_bad_device():
    topo = cirq.LineTopology(8)
    qubits = cirq.LineQubit.range(8)
    circuit = cirq.testing.random_circuit(qubits, n_moments=8, op_density=1.0, random_state=52)
    qp = cg.RandomDevicePlacer()
    with pytest.raises(ValueError, match=r'.*shared_rt_info\.device.*'):
        qp.place_circuit(
            circuit,
            problem_topology=topo,
            shared_rt_info=cg.SharedRuntimeInfo(run_id='1'),
            rs=np.random.RandomState(1),
        )


def test_random_device_placer_small_device():
    topo = cirq.TiltedSquareLattice(3, 3)
    qubits = sorted(topo.nodes_to_gridqubits().values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )
    qp = cg.RandomDevicePlacer()
    with pytest.raises(cg.CouldNotPlaceError):
        qp.place_circuit(
            circuit,
            problem_topology=topo,
            shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=FakeDevice()),
            rs=np.random.RandomState(1),
        )


def test_device_missing_metadata():
    class BadDevice(cirq.Device):
        pass

    topo = cirq.TiltedSquareLattice(3, 3)
    qubits = sorted(topo.nodes_to_gridqubits().values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b)
    )
    qp = cg.RandomDevicePlacer()
    with pytest.raises(ValueError):
        qp.place_circuit(
            circuit,
            problem_topology=topo,
            shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=BadDevice()),
            rs=np.random.RandomState(1),
        )
