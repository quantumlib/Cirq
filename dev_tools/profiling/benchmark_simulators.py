# Copyright 2018 The Cirq Developers
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

"""Tool to benchmarking simulators against a random circuit."""

import argparse
import sys
import timeit

import numpy as np

import cirq

_UNITARY = 'unitary'
_DENSITY = 'density_matrix'


test_device = cirq.google.XmonDevice(
    measurement_duration=cirq.Duration(nanos=1000),
    exp_w_duration=cirq.Duration(nanos=20),
    exp_11_duration=cirq.Duration(nanos=50),
    qubits=[cirq.GridQubit(0, k) for k in range(100)],
)


def simulate(sim_type: str, num_qubits: int, num_gates: int) -> None:
    """"Runs the simulator."""
    circuit = cirq.Circuit(device=test_device)

    for _ in range(num_gates):
        which = np.random.choice(['expz', 'expw', 'exp11'])
        if which == 'expw':
            q1 = cirq.GridQubit(0, np.random.randint(num_qubits))
            circuit.append(
                cirq.PhasedXPowGate(
                    phase_exponent=np.random.random(), exponent=np.random.random()
                ).on(q1)
            )
        elif which == 'expz':
            q1 = cirq.GridQubit(0, np.random.randint(num_qubits))
            circuit.append(cirq.Z(q1) ** np.random.random())
        elif which == 'exp11':
            q1 = cirq.GridQubit(0, np.random.randint(num_qubits - 1))
            q2 = cirq.GridQubit(0, q1.col + 1)
            circuit.append(cirq.CZ(q1, q2) ** np.random.random())
    circuit.append([cirq.measure(cirq.GridQubit(0, np.random.randint(num_qubits)), key='meas')])

    if sim_type == _UNITARY:
        circuit.final_state_vector(initial_state=0)
    elif sim_type == _DENSITY:
        cirq.DensityMatrixSimulator().run(circuit)


def main(
    sim_type: str,
    min_num_qubits: int,
    max_num_qubits: int,
    num_gates: int,
    num_repetitions: int,
    setup: str = 'from __main__ import simulate',
):
    print('num_qubits,seconds per gate')
    for num_qubits in range(min_num_qubits, max_num_qubits + 1):
        command = 'simulate(\'{}\', {}, {})'.format(sim_type, num_qubits, num_gates)
        time = timeit.timeit(command, setup, number=num_repetitions)
        print('{},{}'.format(num_qubits, time / (num_repetitions * num_gates)))


def parse_arguments(args):
    parser = argparse.ArgumentParser('Benchmark a simulator.')
    parser.add_argument(
        '--sim_type',
        choices=[_UNITARY, _DENSITY],
        default=_UNITARY,
        help='Which simulator to benchmark.',
        type=str,
    )
    parser.add_argument(
        '--min_num_qubits', default=4, type=int, help='Minimum number of qubits to benchmark.'
    )
    parser.add_argument(
        '--max_num_qubits', default=26, type=int, help='Maximum number of qubits to benchmark.'
    )
    parser.add_argument(
        '--num_gates', default=100, type=int, help='Number of gates in a single run.'
    )
    parser.add_argument(
        '--num_repetitions', default=10, type=int, help='Number of times to repeat a simulation'
    )
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
