"""Tests for the Quantum Volume benchmarker."""

from unittest.mock import Mock, MagicMock
import io
import numpy as np
from examples.advanced import quantum_volume
import cirq


def test_main_loop():
    """Test that the main loop is able to run without erring."""
    # Keep test from taking a long time by lowering repetitions.
    args = '--num_qubits 5 --depth 5 --num_circuits 1  --routes 3'.split()
    quantum_volume.main(**quantum_volume.parse_arguments(args))


def test_parse_args():
    """Test that an argument string is parsed correctly."""
    args = ('--num_qubits 5 --depth 5 --num_circuits 200 --seed 1234').split()
    kwargs = quantum_volume.parse_arguments(args)
    assert kwargs == {
        'num_qubits': 5,
        'depth': 5,
        'num_circuits': 200,
        'seed': 1234,
        'routes': 30,
    }
