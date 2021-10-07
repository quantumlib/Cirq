"""Tests for the Quantum Volume utilities."""

from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult


class TestDevice(cirq.Device):
    qubits = cirq.GridQubit.rect(5, 5)


def test_generate_model_circuit():
    """Test that a model circuit is randomly generated."""
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1)
    )

    assert len(model_circuit) == 3
    # Ensure there are no measurement gates.
    assert list(model_circuit.findall_operations_with_gate_type(cirq.MeasurementGate)) == []


def test_generate_model_circuit_without_seed():
    """Test that a model circuit is randomly generated without a seed."""
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(3, 3)

    assert len(model_circuit) == 3
    # Ensure there are no measurement gates.
    assert list(model_circuit.findall_operations_with_gate_type(cirq.MeasurementGate)) == []


def test_generate_model_circuit_seed():
    """Test that a model circuit is determined by its seed ."""
    model_circuit_1 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1)
    )
    model_circuit_2 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1)
    )
    model_circuit_3 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(2)
    )

    assert model_circuit_1 == model_circuit_2
    assert model_circuit_2 != model_circuit_3


def test_compute_heavy_set():
    """Test that the heavy set can be computed from a given circuit."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([]),
            cirq.Moment([cirq.X(a), cirq.Y(b)]),
            cirq.Moment([]),
            cirq.Moment([cirq.CNOT(a, c)]),
            cirq.Moment([cirq.Z(a), cirq.H(b)]),
        ]
    )
    assert cirq.contrib.quantum_volume.compute_heavy_set(model_circuit) == [5, 7]


def test_sample_heavy_set():
    """Test that we correctly sample a circuit's heavy set"""

    sampler = Mock(spec=cirq.Simulator)
    # Construct a result that returns "1", "2", "3", "0"
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={'mock': np.array([[0, 1], [1, 0], [1, 1], [0, 0]])},
    )
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(2)))
    compilation_result = CompilationResult(circuit=circuit, mapping={}, parity_map={})
    probability = cirq.contrib.quantum_volume.sample_heavy_set(
        compilation_result, [1, 2, 3], sampler=sampler, repetitions=10
    )
    # The first 3 of our outputs are in the heavy set, and then the rest are
    # not.
    assert probability == 0.75


def test_sample_heavy_set_with_parity():
    """Test that we correctly sample a circuit's heavy set with a parity map"""

    sampler = Mock(spec=cirq.Simulator)
    # Construct a result that returns [1, 0, 1, 0] for the physical qubit
    # measurement, and [0, 1, 1, 0] for the ancilla qubit measurement. The first
    # bitstring "10" is valid and heavy. The second "01" is valid and not
    # heavy. The third and fourth bitstrings "11" and "00" are not valid and
    # dropped.
    result = cirq.Result.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            '0': np.array([[1], [0]]),
            '1': np.array([[0], [1]]),
            '2': np.array([[1], [1]]),
            '3': np.array([[0], [0]]),
        },
    )
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(4)))
    compilation_result = CompilationResult(
        circuit=circuit,
        mapping={q: q for q in cirq.LineQubit.range(4)},
        parity_map={cirq.LineQubit(0): cirq.LineQubit(1), cirq.LineQubit(2): cirq.LineQubit(3)},
    )
    probability = cirq.contrib.quantum_volume.sample_heavy_set(
        compilation_result, [1], sampler=sampler, repetitions=1
    )
    # The first output is in the heavy set. The second one isn't, but it is
    # dropped.
    assert probability == 0.5


def test_compile_circuit_router():
    """Tests that the given router is used."""
    router_mock = MagicMock()
    cirq.contrib.quantum_volume.compile_circuit(
        cirq.Circuit(),
        device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
        router=router_mock,
        routing_attempts=1,
    )
    router_mock.assert_called()


def test_compile_circuit():
    """Tests that we are able to compile a model circuit."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
        ]
    )
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
        compiler=compiler_mock,
        routing_attempts=1,
    )

    assert len(compilation_result.mapping) == 3
    assert cirq.contrib.routing.ops_are_consistent_with_device_graph(
        compilation_result.circuit.all_operations(),
        cirq.contrib.routing.gridqubits_to_graph_device(TestDevice().qubits),
    )
    compiler_mock.assert_called_with(compilation_result.circuit)


def test_compile_circuit_replaces_swaps():
    """Tests that the compiler never sees the SwapPermutationGates from the
    router."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    # Create a circuit that will require some swaps.
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.CNOT(a, b)]),
            cirq.Moment([cirq.CNOT(a, c)]),
            cirq.Moment([cirq.CNOT(b, c)]),
        ]
    )
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
        compiler=compiler_mock,
        routing_attempts=1,
    )

    # Assert that there were some swaps in the result
    compiler_mock.assert_called_with(compilation_result.circuit)
    assert (
        len(
            list(compilation_result.circuit.findall_operations_with_gate_type(cirq.ops.SwapPowGate))
        )
        > 0
    )
    # Assert that there were not SwapPermutations in the result.
    assert (
        len(
            list(
                compilation_result.circuit.findall_operations_with_gate_type(
                    cirq.contrib.acquaintance.SwapPermutationGate
                )
            )
        )
        == 0
    )


def test_compile_circuit_with_readout_correction():
    """Tests that we are able to compile a model circuit with readout error
    correction."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    router_mock = MagicMock(side_effect=lambda circuit, network: ccr.SwapNetwork(circuit, {}))
    a, b, c = cirq.LineQubit.range(3)
    ap, bp, cp = cirq.LineQubit.range(3, 6)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
        ]
    )
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
        compiler=compiler_mock,
        router=router_mock,
        routing_attempts=1,
        add_readout_error_correction=True,
    )

    assert compilation_result.circuit == cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
            cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)]),
            cirq.Moment([cirq.CNOT(a, ap), cirq.CNOT(b, bp), cirq.CNOT(c, cp)]),
            cirq.Moment([cirq.X(a), cirq.X(b), cirq.X(c)]),
        ]
    )


def test_compile_circuit_multiple_routing_attempts():
    """Tests that we make multiple attempts at routing and keep the best one."""
    qubits = cirq.LineQubit.range(3)
    initial_mapping = dict(zip(qubits, qubits))
    more_operations = cirq.Circuit(
        [
            cirq.X.on_each(qubits),
            cirq.Y.on_each(qubits),
        ]
    )
    more_qubits = cirq.Circuit(
        [
            cirq.X.on_each(cirq.LineQubit.range(4)),
        ]
    )
    well_routed = cirq.Circuit(
        [
            cirq.X.on_each(qubits),
        ]
    )
    router_mock = MagicMock(
        side_effect=[
            ccr.SwapNetwork(more_operations, initial_mapping),
            ccr.SwapNetwork(well_routed, initial_mapping),
            ccr.SwapNetwork(more_qubits, initial_mapping),
        ]
    )
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    model_circuit = cirq.Circuit([cirq.X.on_each(qubits)])

    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
        compiler=compiler_mock,
        router=router_mock,
        routing_attempts=3,
    )

    assert compilation_result.mapping == initial_mapping
    assert router_mock.call_count == 3
    compiler_mock.assert_called_with(well_routed)


def test_compile_circuit_no_routing_attempts():
    """Tests that setting no routing attempts throws an error."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit(
        [
            cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
        ]
    )

    with pytest.raises(AssertionError) as e:
        cirq.contrib.quantum_volume.compile_circuit(
            model_circuit,
            device_graph=ccr.gridqubits_to_graph_device(TestDevice().qubits),
            routing_attempts=0,
        )
    assert e.match('Unable to get routing for circuit')


def test_calculate_quantum_volume_result():
    """Test that running the main loop returns the desired result"""
    results = cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=3,
        depth=3,
        num_circuits=1,
        device_graph=ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3)),
        samplers=[cirq.Simulator()],
        routing_attempts=2,
        random_state=1,
    )

    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(3, 3, random_state=1)
    assert len(results) == 1
    assert results[0].model_circuit == model_circuit
    assert results[0].heavy_set == cirq.contrib.quantum_volume.compute_heavy_set(model_circuit)
    # Ensure that calling to_json on the results does not err.
    buffer = io.StringIO()
    cirq.to_json(results, buffer)


def test_calculate_quantum_volume_result_with_device_graph():
    """Test that running the main loop routes the circuit onto the given device
    graph"""
    device_qubits = [cirq.GridQubit(i, j) for i in range(2) for j in range(3)]

    results = cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=3,
        depth=3,
        num_circuits=1,
        device_graph=ccr.gridqubits_to_graph_device(device_qubits),
        samplers=[cirq.Simulator()],
        routing_attempts=2,
        random_state=1,
    )

    assert len(results) == 1
    assert ccr.ops_are_consistent_with_device_graph(
        results[0].compiled_circuit.all_operations(), ccr.get_grid_device_graph(2, 3)
    )


def test_calculate_quantum_volume_loop():
    """Test that calculate_quantum_volume is able to run without erring."""
    # Keep test from taking a long time by lowering circuits and routing
    # attempts.
    cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=5,
        depth=5,
        num_circuits=1,
        routing_attempts=2,
        random_state=1,
        device_graph=ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3)),
        samplers=[cirq.Simulator()],
    )


def test_calculate_quantum_volume_loop_with_readout_correction():
    """Test that calculate_quantum_volume is able to run without erring with
    readout error correction."""
    # Keep test from taking a long time by lowering circuits and routing
    # attempts.
    cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=4,
        depth=4,
        num_circuits=1,
        routing_attempts=2,
        random_state=1,
        device_graph=ccr.gridqubits_to_graph_device(cirq.GridQubit.rect(3, 3)),
        samplers=[cirq.Simulator()],
        add_readout_error_correction=True,
    )
