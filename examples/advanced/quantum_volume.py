# pylint: disable=wrong-or-nonexistent-copyright-notice
"""Tool to run the Quantum Volume benchmark defined by IBM in
https://arxiv.org/abs/1811.12926. By default, this runs on the Bristlecone
device.

Usage: python examples/advanced/quantum_volume.py \
         --num_qubits=4 --depth=4 --num_circuits=1 [--seed=int]

Output:
    When run, this program will return a QuantumVolumeResult object containing
    the computed model circuits, their heavy sets, their compiled circuits, and
    the results of running each sampler on the compiled circuits.

    This program it also prints the Heavy Set of result values that represent
    the bit-strings produced by a randomly-generated model circuit (Example: [1,
    5, 7]), and the HOG probability for each given sampler when run on the
    device.

"""

import argparse
import sys

import cirq
import cirq_google
from cirq.contrib.quantum_volume import calculate_quantum_volume
from cirq.contrib import routing


def main(*, num_qubits: int, depth: int, num_circuits: int, seed: int, routes: int):
    """Run the quantum volume algorithm with a preset configuration.

    See the calculate_quantum_volume documentation for more details.

    Args:
        num_qubits: Pass-through to calculate_quantum_volume.
        depth: Pass-through to calculate_quantum_volume
        num_circuits: Pass-through to calculate_quantum_volume
        seed: Pass-through to calculate_quantum_volume
        routes: Pass-through to calculate_quantum_volume as routing_attempts

    Returns: Pass-through from calculate_quantum_volume.
    """
    device = cirq_google.Sycamore
    # compiler = lambda circuit: cirq_google.optimized_for_xmon(circuit=circuit)
    compiler = lambda circuit: cirq.optimize_for_target_gateset(
        circuit, gateset=cirq.CZTargetGateset()
    )
    noisy = cirq.DensityMatrixSimulator(
        noise=cirq.ConstantQubitNoiseModel(qubit_noise_gate=cirq.DepolarizingChannel(p=0.005))
    )
    calculate_quantum_volume(
        num_qubits=num_qubits,
        depth=depth,
        num_circuits=num_circuits,
        random_state=seed,
        device_graph=routing.gridqubits_to_graph_device(device.metadata.qubit_set),
        samplers=[cirq.Simulator(), noisy],
        routing_attempts=routes,
        compiler=compiler,
    )


def parse_arguments(args):
    """Helper function that parses the given arguments."""
    parser = argparse.ArgumentParser('Quantum volume benchmark.')
    parser.add_argument(
        '--num_qubits', default=4, type=int, help='The number of circuit qubits to benchmark.'
    )
    parser.add_argument('--depth', default=4, type=int, help='SU(4) circuit depth.')
    parser.add_argument(
        '--routes', default=30, type=int, help='Number of different qubit routes to try'
    )
    parser.add_argument(
        '--seed', default=None, type=int, help='Seed for the Random Number Generator.'
    )
    parser.add_argument(
        '--num_circuits',
        default=100,
        type=int,
        help='The number of random circuits to run on the quantum computer.'
        ' According to the source paper, this should be at least 100.',
    )
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
