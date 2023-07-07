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
import itertools

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


def _all_offset_placements(device_graph, offset=(4, 2), min_sidelength=2, max_sidelength=5):
    # Generate candidate tilted square lattice topologies
    sidelens = list(itertools.product(range(min_sidelength, max_sidelength + 1), repeat=2))
    topos = [cirq.TiltedSquareLattice(width, height) for width, height in sidelens]

    # Make placements using TiltedSquareLattice.nodes_to_gridqubits offset parameter
    placements = {topo: topo.nodes_to_gridqubits(offset=offset) for topo in topos}

    # Only allow placements that are valid on the device graph
    placements = {
        topo: mapping
        for topo, mapping in placements.items()
        if cirq.is_valid_placement(device_graph, topo.graph, mapping)
    }
    return placements


def test_hardcoded_qubit_placer():

    rainbow_record = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    rainbow_device = rainbow_record.get_device()
    rainbow_graph = rainbow_device.metadata.nx_graph
    hardcoded = cg.HardcodedQubitPlacer(_all_offset_placements(rainbow_graph))

    topo = cirq.TiltedSquareLattice(3, 2)
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
        qubits=sorted(topo.nodes_as_gridqubits()), depth=4
    )
    shared_rt_info = cg.SharedRuntimeInfo(run_id='example', device=rainbow_device)

    rs = np.random.RandomState(10)
    placed_c, placement = hardcoded.place_circuit(
        circuit, problem_topology=topo, shared_rt_info=shared_rt_info, rs=rs
    )
    cirq.is_valid_placement(rainbow_graph, topo.graph, placement)
    assert isinstance(placed_c, cirq.FrozenCircuit)


def test_hqp_missing_placement():
    hqp = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})

    circuit = cirq.testing.random_circuit(cirq.LineQubit.range(5), n_moments=2, op_density=1)
    shared_rt_info = cg.SharedRuntimeInfo(run_id='example')
    rs = np.random.RandomState(10)
    placed_c, _ = hqp.place_circuit(
        circuit, problem_topology=cirq.LineTopology(5), shared_rt_info=shared_rt_info, rs=rs
    )
    assert isinstance(placed_c, cirq.AbstractCircuit)

    circuit = cirq.testing.random_circuit(cirq.LineQubit.range(6), n_moments=2, op_density=1)
    with pytest.raises(cg.CouldNotPlaceError):
        hqp.place_circuit(
            circuit, problem_topology=cirq.LineTopology(6), shared_rt_info=shared_rt_info, rs=rs
        )


def test_hqp_equality():
    hqp = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})
    hqp2 = cg.HardcodedQubitPlacer({cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(5)))})
    assert hqp == hqp2
    cirq.testing.assert_equivalent_repr(hqp, global_vals={'cirq_google': cg})

    hqp3 = cg.HardcodedQubitPlacer(
        {cirq.LineTopology(5): dict(enumerate(cirq.LineQubit.range(1, 5 + 1)))}
    )
    assert hqp != hqp3
