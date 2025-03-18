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

import pytest
import numpy as np
import cirq
from cirq.experiments.benchmarking import parallel_xeb as xeb
import cirq.experiments.random_quantum_circuit_generation as rqcg

_QUBITS = cirq.LineQubit.range(2)
_CIRCUIT_TEMPLATES = [
    cirq.Circuit(cirq.X.on_each(_QUBITS), cirq.CZ(*_QUBITS), cirq.Y.on_each(_QUBITS)),
    cirq.Circuit(cirq.Y.on_each(_QUBITS), cirq.CX(*_QUBITS), cirq.Z.on_each(_QUBITS)),
    cirq.Circuit(cirq.Z.on_each(_QUBITS), cirq.CZ(*_QUBITS), cirq.X.on_each(_QUBITS)),
]
_PAIRS = [(cirq.q(0, 0), cirq.q(0, 1)), (cirq.q(0, 2), cirq.q(0, 3)), (cirq.q(0, 4), cirq.q(0, 5))]


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
            narrow_template_indicies=permutation,
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
            narrow_template_indicies=[1, 2, 0],
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
            narrow_template_indicies=[1, 2, 0],
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
            narrow_template_indicies=[1, 2, 0],
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
            narrow_template_indicies=[1, 2, 0],
        )
    ]


def test_simulate_circuit():
    sim = cirq.Simulator()
    circuit_id = 123
    circuit = cirq.Circuit.from_moments(
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
        cirq.CNOT(*_QUBITS),
        cirq.X.on_each(_QUBITS),
    )

    circuit_id, result = xeb.simulate_circuit(sim, circuit, [1, 3], circuit_id)
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
    result = xeb.sample_all_circuits(cirq.Simulator(), circuits=wide_circuits, repetitions=10)
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


def test_estimate_fidilties():
    sampling_result = [{str(_PAIRS[0]): np.array([0.15, 0.15, 0.35, 0.35])}]

    simulation_results = [[np.array([0.1, 0.2, 0.3, 0.4])]]

    result = xeb.estimate_fidilties(
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
                narrow_template_indicies=(0,),
                pairs=_PAIRS[:1],
            )
        ],
    )

    assert result == [
        xeb.XEBFidelity(
            pair=_PAIRS[0],
            cycle_depth=1,
            fidelity=pytest.approx(0.8),
            fidelity_variance=pytest.approx(25.84),
        )
    ]


def test_estimate_fidilties_with_dict_target():
    sampling_result = [{str(_PAIRS[0]): np.array([0.15, 0.15, 0.35, 0.35])}]

    simulation_results = {_PAIRS[0]: [[np.array([0.1, 0.2, 0.3, 0.4])]]}

    result = xeb.estimate_fidilties(
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
                narrow_template_indicies=(0,),
                pairs=_PAIRS[:1],
            )
        ],
    )

    assert result == [
        xeb.XEBFidelity(
            pair=_PAIRS[0],
            cycle_depth=1,
            fidelity=pytest.approx(0.8),
            fidelity_variance=pytest.approx(25.84),
        )
    ]


@pytest.mark.parametrize('target', [cirq.CZ, cirq.Circuit(cirq.CZ(*_QUBITS))])
@pytest.mark.parametrize('pairs', [_PAIRS[:1], _PAIRS[:2]])
def test_parallel_two_qubit_xeb(target, pairs):
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03))
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        pairs=pairs,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    np.testing.assert_allclose(result.fidelities.layer_fid, 0.9, atol=0.2)


def test_parallel_two_qubit_xeb_with_dict_target():
    target = {p: cirq.Circuit(cirq.CZ(*_QUBITS)) for p in _PAIRS}
    sampler = cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.03))
    result = xeb.parallel_two_qubit_xeb(
        sampler=sampler,
        target=target,
        parameters=xeb.XEBParameters(
            n_circuits=10, n_combinations=10, n_repetitions=10, cycle_depths=range(1, 10, 2)
        ),
    )
    np.testing.assert_allclose(result.fidelities.layer_fid, 0.9, atol=0.2)
