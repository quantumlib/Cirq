"""Tests for the Quantum Volume benchmarker."""

from unittest.mock import Mock, MagicMock
import numpy as np
from examples.advanced import quantum_volume
import cirq


def test_generate_model_circuit():
    """Test that a model circuit is randomly generated."""
    model_circuit = quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))

    assert len(model_circuit) == 24
    # Ensure there are no measurement gates.
    assert list(
        model_circuit.findall_operations_with_gate_type(
            cirq.MeasurementGate)) == []


def test_generate_model_circuit_without_seed():
    """Test that a model circuit is randomly generated without a seed."""
    model_circuit = quantum_volume.generate_model_circuit(3, 3)

    assert len(model_circuit) == 24
    # Ensure there are no measurement gates.
    assert list(
        model_circuit.findall_operations_with_gate_type(
            cirq.MeasurementGate)) == []


def test_generate_model_circuit_seed():
    """Test that a model circuit is determined by its seed ."""
    model_circuit_1 = quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))
    model_circuit_2 = quantum_volume.generate_model_circuit(
        3, 3, random_state=np.random.RandomState(1))
    model_circuit_3 = quantum_volume.generate_model_circuit(
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
    assert quantum_volume.compute_heavy_set(model_circuit) == [5, 7]


def test_sample_heavy_set():
    """Test that we correctly sample a circuit's heavy set"""

    sampler = Mock(spec=cirq.Simulator)
    # Construct a result that returns "1", "2", "3", and then "0" indefinitely
    result = cirq.TrialResult.from_single_parameter_set(
        params=cirq.ParamResolver({}),
        measurements={'mock': np.array([[0, 1], [1, 0], [1, 1], [0, 0]])})
    sampler.run = MagicMock(return_value=result)
    circuit = cirq.Circuit.from_ops(cirq.measure(*cirq.LineQubit.range(2)))

    probability = quantum_volume.sample_heavy_set(circuit, [1, 2, 3],
                                                  sampler=sampler,
                                                  repetitions=1000)
    # The first 3 of our outputs are in the heavy set, and then the rest are
    # not.
    assert probability == .003


def test_main_loop():
    """Test that the main loop is able to run without erring."""
    # Keep test from taking a long time by lowering repetitions.
    args = '--num_qubits 5 --depth 5 --num_repetitions 1'.split()
    quantum_volume.main(**quantum_volume.parse_arguments(args))


def test_parse_args():
    """Test that an argument string is parsed correctly."""
    args = (
        '--num_qubits 5 --depth 5 --num_repetitions 200 --seed 1234').split()
    kwargs = quantum_volume.parse_arguments(args)
    assert kwargs == {
        'num_qubits': 5,
        'depth': 5,
        'num_repetitions': 200,
        'seed': 1234,
    }
