"""Tests for the Quantum Volume utilities."""

from unittest.mock import Mock, MagicMock
import io
import numpy as np
import pytest
import cirq
import cirq.contrib.routing as ccr
from cirq.contrib.quantum_volume import CompilationResult


def test_generate_model_circuit():
    """Test that a model circuit is randomly generated."""
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))

    assert len(model_circuit) == 3
    # Ensure there are no measurement gates.
    assert list(
        model_circuit.findall_operations_with_gate_type(
            cirq.MeasurementGate)) == []


def test_generate_model_circuit_without_seed():
    """Test that a model circuit is randomly generated without a seed."""
    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(3, 3)

    assert len(model_circuit) == 3
    # Ensure there are no measurement gates.
    assert list(
        model_circuit.findall_operations_with_gate_type(
            cirq.MeasurementGate)) == []


def test_generate_model_circuit_seed():
    """Test that a model circuit is determined by its seed ."""
    model_circuit_1 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))
    model_circuit_2 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))
    model_circuit_3 = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(2))

    assert model_circuit_1 == model_circuit_2
    assert model_circuit_2 != model_circuit_3


def test_compute_heavy_set():
    """Test that the heavy set can be computed from a given circuit."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([
        cirq.Moment([]),
        cirq.Moment([cirq.X(a), cirq.Y(b)]),
        cirq.Moment([]),
        cirq.Moment([cirq.CNOT(a, c)]),
        cirq.Moment([cirq.Z(a), cirq.H(b)])
    ])
    assert cirq.contrib.quantum_volume.compute_heavy_set(model_circuit) == [
        5, 7
    ]


def test_sample_heavy_set():
    """Test that we correctly sample a circuit's heavy set"""

    sampler = Mock(spec=cirq.Simulator)
    # Construct a result that returns "1", "2", "3", "0"
    result = cirq.TrialResult.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={'mock': np.array([[0, 1], [1, 0], [1, 1], [0, 0]])})
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(2)))
    compilation_result = CompilationResult(swap_network=ccr.SwapNetwork(
        circuit, {}),
                                           parity_map={})
    probability = cirq.contrib.quantum_volume.sample_heavy_set(
        compilation_result, [1, 2, 3], sampler=sampler, repetitions=10)
    # The first 3 of our outputs are in the heavy set, and then the rest are
    # not.
    assert probability == .75


def test_sample_heavy_set_with_parity():
    """Test that we correctly sample a circuit's heavy set with a parity map"""

    sampler = Mock(spec=cirq.Simulator)
    # Construct a result that returns [1, 0, 1, 0] for the physical qubit
    # measurement, and [0, 1, 1, 0] for the ancilla qubit measurement. The first
    # bitstring "10" is valid and heavy. The second "01" is valid and not
    # heavy. The third and fourth bitstraings "11" and "00" are not valid and
    # dropped.
    result = cirq.TrialResult.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={
            '0': np.array([[1], [0]]),
            '1': np.array([[0], [1]]),
            '2': np.array([[1], [1]]),
            '3': np.array([[0], [0]])
        })
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit(cirq.measure(*cirq.LineQubit.range(4)))
    compilation_result = CompilationResult(
        swap_network=ccr.SwapNetwork(circuit,
                                     {q: q for q in cirq.LineQubit.range(4)}),
        parity_map={
            cirq.LineQubit(0): cirq.LineQubit(1),
            cirq.LineQubit(2): cirq.LineQubit(3)
        })
    probability = cirq.contrib.quantum_volume.sample_heavy_set(
        compilation_result, [1], sampler=sampler, repetitions=1)
    # The first output is in the heavy set. The second one isn't, but it is
    # dropped.
    assert probability == .5


def test_compile_circuit_router():
    """Tests that the given router is used."""
    router_mock = MagicMock()
    cirq.contrib.quantum_volume.compile_circuit(cirq.Circuit(),
                                                device=cirq.google.Bristlecone,
                                                router=router_mock,
                                                routing_attempts=1)
    router_mock.assert_called()


def test_compile_circuit():
    """Tests that we are able to compile a model circuit."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
    ])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device=cirq.google.Bristlecone,
        compiler=compiler_mock,
        routing_attempts=1)

    assert len(compilation_result.swap_network.final_mapping()) == 3
    assert cirq.contrib.routing.ops_are_consistent_with_device_graph(
        compilation_result.swap_network.circuit.all_operations(),
        cirq.contrib.routing.xmon_device_to_graph(cirq.google.Bristlecone))
    compiler_mock.assert_called_with(compilation_result.swap_network.circuit)


def test_compile_circuit_with_readout_correction():
    """Tests that we are able to compile a model circuit."""
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
    ])
    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device=cirq.google.Bristlecone,
        compiler=compiler_mock,
        routing_attempts=1,
        add_readout_error_correction=True)

    assert len(compilation_result.swap_network.final_mapping()) == 6
    assert compilation_result.parity_map == {
        cirq.LineQubit(0): cirq.LineQubit(3),
        cirq.LineQubit(1): cirq.LineQubit(4),
        cirq.LineQubit(2): cirq.LineQubit(5)
    }
    assert cirq.contrib.routing.ops_are_consistent_with_device_graph(
        compilation_result.swap_network.circuit.all_operations(),
        cirq.contrib.routing.xmon_device_to_graph(cirq.google.Bristlecone))
    compiler_mock.assert_called_with(compilation_result.swap_network.circuit)


def test_compile_circuit_multiple_routing_attempts():
    """Tests that we make multiple attempts at r
    outing and keep the best one."""
    qubits = cirq.LineQubit.range(3)
    initial_mapping = dict(zip(qubits, qubits))
    badly_routed = cirq.Circuit([
        cirq.X.on_each(qubits),
        cirq.Y.on_each(qubits),
    ])
    well_routed = cirq.Circuit([
        cirq.X.on_each(qubits),
    ])
    router_mock = MagicMock(side_effect=[
        ccr.SwapNetwork(badly_routed, initial_mapping),
        ccr.SwapNetwork(well_routed, initial_mapping),
    ])
    compiler_mock = MagicMock(side_effect=lambda circuit: circuit)
    model_circuit = cirq.Circuit([cirq.X.on_each(qubits)])

    compilation_result = cirq.contrib.quantum_volume.compile_circuit(
        model_circuit,
        device=cirq.google.Bristlecone,
        compiler=compiler_mock,
        router=router_mock,
        routing_attempts=2)

    assert compilation_result.swap_network.final_mapping() == initial_mapping
    assert router_mock.call_count == 2
    compiler_mock.assert_called_with(well_routed)


def test_compile_circuit_no_routing_attempts():
    """Tests that setting no routing attempts throws an error."""
    a, b, c = cirq.LineQubit.range(3)
    model_circuit = cirq.Circuit([
        cirq.Moment([cirq.X(a), cirq.Y(b), cirq.Z(c)]),
    ])

    with pytest.raises(AssertionError) as e:
        cirq.contrib.quantum_volume.compile_circuit(
            model_circuit, device=cirq.google.Bristlecone, routing_attempts=0)
    assert e.match('Unable to get routing for circuit')


def test_calculate_quantum_volume_result():
    """Test that running the main loop returns the desired result"""
    results = cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=3,
        depth=3,
        num_circuits=1,
        device=cirq.google.Bristlecone,
        samplers=[cirq.Simulator()],
        routing_attempts=2,
        random_state=1,
    )

    model_circuit = cirq.contrib.quantum_volume.generate_model_circuit(
        3, 3, random_state=1)
    assert len(results) == 1
    assert results[0].model_circuit == model_circuit
    assert results[
        0].heavy_set == cirq.contrib.quantum_volume.compute_heavy_set(
            model_circuit)
    # Ensure that calling to_json on the results does not err.
    buffer = io.StringIO()
    cirq.to_json(results, buffer)


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
        device=cirq.google.Bristlecone,
        samplers=[cirq.Simulator()])


def test_calculate_quantum_volume_loop_with_readout_correction():
    """Test that calculate_quantum_volume is able to run without erring with
       readout error correction."""
    # Keep test from taking a long time by lowering circuits and routing
    # attempts.
    cirq.contrib.quantum_volume.calculate_quantum_volume(
        num_qubits=5,
        depth=5,
        num_circuits=1,
        routing_attempts=2,
        random_state=1,
        device=cirq.google.Bristlecone,
        samplers=[cirq.Simulator()],
        add_readout_error_correction=True)
