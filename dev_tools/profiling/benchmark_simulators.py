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
from cirq.google import ExpWGate, ExpZGate, XmonOptions, XmonSimulator


_XMON = 'xmon'
_UNITARY = 'unitary'


def simulate(
    sim_type: str,
    num_qubits: int,
    num_gates: int,
    num_prefix_qubits: int = 0,
    use_processes: bool = False) -> None:
    """"Runs the simulator."""
    circuit = cirq.Circuit()
    for _ in range(num_gates):
        which = np.random.choice(['expz', 'expw', 'exp11'])
        if which == 'expw':
            circuit.append(ExpWGate(axis_half_turns=np.random.random(),
                                    half_turns=np.random.random()).on(
                np.random.randint(num_qubits)),
                strategy=cirq.InsertStrategy.EARLIEST)
        elif which == 'expz':
            circuit.append(ExpZGate(half_turns=np.random.random()).on(
                np.random.randint(num_qubits)),
                strategy=cirq.InsertStrategy.EARLIEST)
        elif which == 'exp11':
            q1, q2 = np.random.choice(num_qubits, 2, replace=False)
            circuit.append(cirq.CZ(q1, q2)**np.random.random(),
                           strategy=cirq.InsertStrategy.EARLIEST)

    if sim_type == _XMON:
        XmonSimulator(XmonOptions(num_shards=2 ** num_prefix_qubits,
                                  use_processes=use_processes)).run(circuit)
    elif sim_type == _UNITARY:
        circuit.apply_unitary_effect_to_state(initial_state=0)


def main(
    sim_type: str,
    min_num_qubits: int,
    max_num_qubits: int,
    num_gates: int,
    num_prefix_qubits: int,
    use_processes: bool,
    num_repetitions: int,
    setup: str= 'from __main__ import simulate'):
    print('num_qubits,seconds per gate')
    for num_qubits in range(min_num_qubits, max_num_qubits + 1):
        command = 'simulate(\'{}\', {}, {}, {}, {})'.format(sim_type,
                                                            num_qubits,
                                                            num_gates,
                                                            num_prefix_qubits,
                                                            use_processes)
        time = timeit.timeit(command, setup, number=num_repetitions)
        print('{},{}'.format(num_qubits, time / (num_repetitions * num_gates)))



def parse_arguments(args):
    parser = argparse.ArgumentParser('Benchmark a simulator.')
    parser.add_argument('--sim_type', choices=[_XMON, _UNITARY], default=_XMON,
                        help='Which simulator to benchmark.', type=str)
    parser.add_argument('--min_num_qubits', default=4, type=int,
                        help='Minimum number of qubits to benchmark.')
    parser.add_argument('--max_num_qubits', default=26, type=int,
                        help='Maximum number of qubits to benchmark.')
    parser.add_argument('--num_gates', default=100, type=int,
                        help='Number of gates in a single run.')
    parser.add_argument('--num_repetitions', default=10, type=int,
                        help='Number of times to repeat a simulation')
    parser.add_argument('--num_prefix_qubits', default=2, type=int,
                        help='Used for sharded simulators, the number of '
                             'shards is 2 raised to this number.')
    parser.add_argument('--use_processes', default=False,
                        action='store_true',
                        help='Whether or not to use multiprocessing '
                             '(otherwise uses threads).')
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
