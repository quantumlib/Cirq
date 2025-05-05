# Copyright 2025 The Cirq Developers
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
from concurrent import futures
from typing import Iterator

import networkx as nx
import numpy as np
import pytest

import cirq
import cirq.experiments.random_quantum_circuit_generation as rqcg
from cirq.experiments.benchmarking import parallel_xeb as xeb

_QUBITS = cirq.LineQubit.range(2)
_CIRCUIT_TEMPLATES = [
    cirq.Circuit(cirq.X.on_each(_QUBITS), cirq.CZ(*_QUBITS), cirq.Y.on_each(_QUBITS)),
    cirq.Circuit(cirq.Y.on_each(_QUBITS), cirq.CX(*_QUBITS), cirq.Z.on_each(_QUBITS)),
    cirq.Circuit(cirq.Z.on_each(_QUBITS), cirq.CZ(*_QUBITS), cirq.X.on_each(_QUBITS)),
]
_PAIRS = ((cirq.q(0, 0), cirq.q(0, 1)), (cirq.q(0, 2), cirq.q(0, 3)), (cirq.q(0, 4), cirq.q(0, 5)))


class TestXEBWideCircuitInfo:

    def test_from_circuit(self):
        permutation = [1, 2, 0]
        target = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.CNOT(*_QUBITS)))
        wide_circuit = xeb.XEBWideCircuitInfo.from_narrow_circuits(
            _CIRCUIT_TEMPLATES, permutation=permutation, pairs=_PAIRS, target=target
        )
        assert wide_circuit == xeb.XEBWideCircuitInfo(
            wide_circuit=cirq.Circuit.from_moments(
                [
                    cirq.Y.on_each(*_PAIRS[0]),
                    cirq.Z.on_each(*_PAIRS[1]),
                    cirq.X.on_each(*_PAIRS[2]),
                ],
                cirq.CircuitOperation(
                    cirq.FrozenCircuit(
                        cirq.CNOT(*_PAIRS[0]), cirq.CNOT(*_PAIRS[1]), cirq.CNOT(*_PAIRS[2])
                    )
                ),
                [
                    cirq.Z.on_each(*_PAIRS[0]),
                    cirq.X.on_each(*_PAIRS[1]),
                    cirq.Y.on_each(*_PAIRS[2]),
                ],
            ),
            pairs=_PAIRS,
            narrow_template_indices=permutation,
        )

    def test_sliced_circuit(self):
        wid_circuit = xeb.XEBWideCircuitInfo(
            wide_circuit=cirq.Circuit.from_moments(
                [
                    cirq.Y.on_each(*_PAIRS[0]),
                    cirq.Z.on_each(*_PAIRS[1]),
                    cirq.X.on_each(*_PAIRS[2]),
                ],
                cirq.CircuitOperation(
                    cirq.FrozenCircuit(
                        cirq.CNOT(*_PAIRS[0]), cirq.CNOT(*_PAIRS[1]), cirq.CNOT(*_PAIRS[2])
                    )
                ),
                [
                    cirq.Z.on_each(*_PAIRS[0]),
                    cirq.X.on_each(*_PAIRS[1]),
                    cirq.Y.on_each(*_PAIRS[2]),
                ],
            ),
            pairs=_PAIRS,
            narrow_template_indices=[1, 2, 0],
        )
        sliced_circuit = xeb.XEBWideCircuitInfo(
            wide_circuit=cirq.Circuit.from_moments(
                [
                    cirq.Y.on_each(*_PAIRS[0]),
                    cirq.Z.on_each(*_PAIRS[1]),
                    cirq.X.on_each(*_PAIRS[2]),
                ],
                cirq.CircuitOperation(
                    cirq.FrozenCircuit(
                        cirq.CNOT(*_PAIRS[0]), cirq.CNOT(*_PAIRS[1]), cirq.CNOT(*_PAIRS[2])
                    )
                ),
                [
                    cirq.Z.on_each(*_PAIRS[0]),
                    cirq.X.on_each(*_PAIRS[1]),
                    cirq.Y.on_each(*_PAIRS[2]),
                ],
                [cirq.measure(p, key=str(p)) for p in _PAIRS],
            ),
            pairs=_PAIRS,
            narrow_template_indices=[1, 2, 0],
            cycle_depth=1,
        )

        assert wid_circuit.sliced_circuits([1]) == [sliced_circuit]


def test_create_combination_circuits():
    wide_circuits_info = xeb.create_combination_circuits(
        _CIRCUIT_TEMPLATES,
        [rqcg.CircuitLibraryCombination(layer=None, combinations=[[1, 2, 0]], pairs=_PAIRS)],
        target=cirq.CNOT(*_QUBITS),
    )

    assert wide_circuits_info == [
        xeb.XEBWideCircuitInfo(
            wide_circuit=cirq.Circuit.from_moments(
                [
                    cirq.Y.on_each(*_PAIRS[0]),
                    cirq.Z.on_each(*_PAIRS[1]),
                    cirq.X.on_each(*_PAIRS[2]),
                ],
                [cirq.CNOT(*_PAIRS[0]), cirq.CNOT(*_PAIRS[1]), cirq.CNOT(*_PAIRS[2])],
                [
                    cirq.Z.on_each(*_PAIRS[0]),
                    cirq.X.on_each(*_PAIRS[1]),
                    cirq.Y.on_each(*_PAIRS[2]),
                ],
            ),
            pairs=_PAIRS,
            narrow_template_indices=[1, 2, 0],
        )
    ]


def test_create_combination_circuits_with_target_dict():
    wide_circuits_info = xeb.create_combination_circuits(
        _CIRCUIT_TEMPLATES,
        [rqcg.CircuitLibraryCombination(layer=None, combinations=[[1, 2, 0]], pairs=_PAIRS)],
        target={_PAIRS[0]: cirq.CNOT(*_QUBITS), _PAIRS[1]: cirq.CZ(*_QUBITS)},
    )

    # Wide circuit is created for the qubit pairs in the intersection between target.keys()
    # and combination.pairs
    assert wide_circuits_info == [
        xeb.XEBWideCircuitInfo(
            wide_circuit=cirq.Circuit.from_moments(
                [cirq.Y.on_each(*_PAIRS[0]), cirq.Z.on_each(*_PAIRS[1])],
                [cirq.CNOT(*_PAIRS[0]), cirq.CZ(*_PAIRS[1])],
                [cirq.Z.on_each(*_PAIRS[0]), cirq.X.on_each(*_PAIRS[1])],
            ),
            pairs=_PAIRS,
            narrow_template_indices=[1, 2, 0],
        )
    ]


def test_simulate_circuit():
    sim = cirq.Simulator(seed=0)
    circuit = cirq.Circuit.from_moments(
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
    )

    result = xeb.simulate_circuit(sim, circuit, [1, 3])
    np.testing.assert_allclose(result, [[0, 1, 0, 0], [1, 0, 0, 0]])  # |01>  # |00>


def test_simulate_circuit_library():
    circuit_templates = [
        cirq.Circuit.from_moments(
            cirq.X.on_each(_QUBITS),
            cirq.CNOT(*_QUBITS),
            cirq.X.on_each(_QUBITS),
            cirq.CNOT(*_QUBITS),
            cirq.X.on_each(_QUBITS),
            cirq.CNOT(*_QUBITS),
            cirq.X.on_each(_QUBITS),
        )
    ]

    result = xeb.simulate_circuit_library(
        circuit_templates=circuit_templates, target_or_dict=cirq.CNOT(*_QUBITS), cycle_depths=(1, 3)
    )
    np.testing.assert_allclose(result, [[[0, 1, 0, 0], [1, 0, 0, 0]]])  # |01>  # |00>


def test_simulate_circuit_library_with_target_dict():
    circuit_templates = [
        cirq.Circuit.from_moments(
            [cirq.H(_QUBITS[0]), cirq.X(_QUBITS[1])],
            cirq.CNOT(*_QUBITS),
            [cirq.H(_QUBITS[0]), cirq.X(_QUBITS[1])],
            cirq.CNOT(*_QUBITS),
            [cirq.H(_QUBITS[0]), cirq.X(_QUBITS[1])],
            cirq.CNOT(*_QUBITS),
            [cirq.H(_QUBITS[0]), cirq.X(_QUBITS[1])],
        )
    ]

    result = xeb.simulate_circuit_library(
        circuit_templates=circuit_templates,
        target_or_dict={_PAIRS[0]: cirq.CNOT(*_QUBITS), _PAIRS[1]: cirq.CZ(*_QUBITS)},
        cycle_depths=(1, 3),
    )

    # First pair result.
    np.testing.assert_allclose(result[_PAIRS[0]], [[[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0]]])

    # Second pair result.
    np.testing.assert_allclose(result[_PAIRS[1]], [[[0, 0, 1, 0], [1, 0, 0, 0]]])  # |10>  # |00>


def test_sample_all_circuits():
    wide_circuits = [
        cirq.Circuit.from_moments(
            [cirq.Y.on_each(*_PAIRS[0]), cirq.Z.on_each(*_PAIRS[1]), cirq.X.on_each(*_PAIRS[2])],
            cirq.CircuitOperation(
                cirq.FrozenCircuit(
                    cirq.CNOT(*_PAIRS[0]), cirq.CNOT(*_PAIRS[1]), cirq.CNOT(*_PAIRS[2])
                )
            ),
            [cirq.Z.on_each(*_PAIRS[0]), cirq.X.on_each(*_PAIRS[1]), cirq.Y.on_each(*_PAIRS[2])],
            [cirq.measure(p, key=str(p)) for p in _PAIRS],
        )
    ]
    result = xeb.sample_all_circuits(cirq.Simulator(seed=0), circuits=wide_circuits, repetitions=10)
    assert len(result) == len(wide_circuits)
    assert result[0].keys() == {str(p) for p in _PAIRS}
    np.testing.assert_allclose(
        result[0]['(cirq.GridQubit(0, 0), cirq.GridQubit(0, 1))'], [0, 0, 1, 0]
    )  # |10>
    np.testing.assert_allclose(
        result[0]['(cirq.GridQubit(0, 2), cirq.GridQubit(0, 3))'], [0, 0, 0, 1]
    )  # |11>
    np.testing.assert_allclose(
        result[0]['(cirq.GridQubit(0, 4), cirq.GridQubit(0, 5))'], [0, 1, 0, 0]
    )  # |01>


def test_estimate_fidelities():
    sampling_result = [{str(_PAIRS[0]): np.array([0.15, 0.15, 0.35, 0.35])}]

    simulation_results = [[np.array([0.1, 0.2, 0.3, 0.4])]]

    result = xeb.estimate_fidelities(
        sampling_results=sampling_result,
        simulation_results=simulation_results,
        cycle_depths=(1,),
        num_templates=1,
        pairs=_PAIRS[:1],
        wide_circuits_info=[
            xeb.XEBWideCircuitInfo(
                wide_circuit=cirq.Circuit.from_moments(
                    [cirq.Y.on_each(*_PAIRS[0])],
                    cirq.CNOT(*_PAIRS[0]),
                    [cirq.Z.on_each(*_PAIRS[0])],
                    cirq.measure(_PAIRS[0], key=str(_PAIRS[0])),
                ),
                cycle_depth=1,
                narrow_template_indices=(0,),
                pairs=_PAIRS[:1],
            )
        ],
    )

    assert result == [
        xeb.XEBFidelity(pair=_PAIRS[0], cycle_depth=1, fidelity=pytest.approx(0.785, abs=2e-4))
    ]


def test_estimate_fidelities_with_dict_target():
    sampling_result = [{str(_PAIRS[0]): np.array([0.15, 0.15, 0.35, 0.35])}]

    simulation_results = {_PAIRS[0]: [[np.array([0.1, 0.2, 0.3, 0.4])]]}

    result = xeb.estimate_fidelities(
        sampling_results=sampling_result,
        simulation_results=simulation_results,
        cycle_depths=(1,),
        num_templates=1,
        pairs=_PAIRS[:1],
        wide_circuits_info=[
            xeb.XEBWideCircuitInfo(
                wide_circuit=cirq.Circuit.from_moments(
                    [cirq.Y.on_each(*_PAIRS[0])],
                    cirq.CNOT(*_PAIRS[0]),
                    [cirq.Z.on_each(*_PAIRS[0])],
                    cirq.measure(_PAIRS[0], key=str(_PAIRS[0])),
                ),
                cycle_depth=1,
                narrow_template_indices=(0,),
                pairs=_PAIRS[:1],
            )
        ],
    )

    assert result == [
        xeb.XEBFidelity(pair=_PAIRS[0], cycle_depth=1, fidelity=pytest.approx(0.785, abs=2e-4))
    ]


def _assert_fidelities_approx_equal(fids, expected: float, atol: float):
    fids = np.asarray(fids).tolist()
    fids.sort(reverse=True)
    fids.pop()  # discard smallest to make the test robust to randomness
    np.testing.assert_allclose(fids, expected, atol=atol)


@pytest.mark.parametrize('target', [cirq.CZ, cirq.Circuit(cirq.CZ(*_QUBITS)), cirq.CZ(*_QUBITS)])
@pytest.mark.parametrize('pairs', [_PAIRS[:1], _PAIRS[:2]])
def test_parallel_two_qubit_xeb(target, pairs):
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03), seed=0)
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        pairs=pairs,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    _assert_fidelities_approx_equal(result.fidelities.layer_fid, 0.9, atol=0.3)


class ExampleDevice(cirq.Device):
    @property
    def metadata(self) -> cirq.DeviceMetadata:
        qubits = cirq.GridQubit.rect(3, 2, 4, 3)
        graph = nx.Graph(
            pair
            for pair in itertools.combinations(qubits, 2)
            if abs(pair[0].row - pair[1].row) + abs(pair[0].col - pair[1].col) == 1
        )
        return cirq.DeviceMetadata(qubits, graph)


class ExampleProcessor:
    def get_device(self):
        return ExampleDevice()


class DensityMatrixSimulatorWithProcessor(cirq.DensityMatrixSimulator):
    @property
    def processor(self):
        return ExampleProcessor()


def test_parallel_two_qubit_xeb_with_device():
    target = cirq.CZ
    sampler = DensityMatrixSimulatorWithProcessor(noise=cirq.depolarize(0.03), seed=0)
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    _assert_fidelities_approx_equal(result.fidelities.layer_fid, 0.9, atol=0.3)
    qubits = cirq.GridQubit.rect(3, 2, 4, 3)
    pairs = tuple(
        pair
        for pair in itertools.combinations(qubits, 2)
        if abs(pair[0].row - pair[1].row) + abs(pair[0].col - pair[1].col) == 1
        and pair[0] < pair[1]
    )
    assert result.all_qubit_pairs == pairs


def test_parallel_two_qubit_xeb_with_dict_target():
    target = {p: cirq.Circuit(cirq.CZ(*_QUBITS)) for p in _PAIRS[:2]}
    target[_PAIRS[2]] = cirq.CZ(*_QUBITS)
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03), seed=0)
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    _assert_fidelities_approx_equal(result.fidelities.layer_fid, 0.9, atol=0.3)
    assert result.all_qubit_pairs == _PAIRS


def test_parallel_two_qubit_xeb_with_ideal_target():
    target = {p: cirq.Circuit(cirq.CZ(*_QUBITS)) for p in _PAIRS[:2]}
    target[_PAIRS[2]] = cirq.CZ(*_QUBITS)
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03), seed=0)
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        ideal_target=cirq.CZ,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    _assert_fidelities_approx_equal(result.fidelities.layer_fid, 0.9, atol=0.3)
    assert result.all_qubit_pairs == _PAIRS


@pytest.fixture
def threading_pool() -> Iterator[futures.Executor]:
    with futures.ThreadPoolExecutor(1) as pool:
        yield pool


def test_parallel_two_qubit_xeb_with_dict_target_and_pool(threading_pool):
    target = {p: cirq.Circuit(cirq.CZ(*_QUBITS)) for p in _PAIRS}
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03), seed=0)
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
        pool=threading_pool,
    )
    _assert_fidelities_approx_equal(result.fidelities.layer_fid, 0.9, atol=0.3)
    assert result.all_qubit_pairs == _PAIRS


def test_parallel_two_qubit_xeb_with_invalid_input_raises():
    with pytest.raises(AssertionError):
        _ = xeb.parallel_two_qubit_xeb(
            sampler=cirq.Simulator(seed=0), target={_PAIRS[0]: cirq.CZ}, pairs=_PAIRS
        )

    with pytest.raises(AssertionError):
        _ = xeb.parallel_two_qubit_xeb(
            sampler=cirq.Simulator(seed=0),
            target=cirq.CZ,
            ideal_target={_PAIRS[0]: cirq.CZ},
            pairs=_PAIRS,
        )
