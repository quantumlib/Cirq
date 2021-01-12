# Copyright 2020 The Cirq Developers
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

"""Tool for benchmarking serialization of large circuits."""

import argparse
import sys
import timeit

import numpy as np

import cirq

_GZIP = 'gzip'
_JSON = 'json'

NUM_QUBITS = 8

SUFFIXES = ['B', 'kB', 'MB', 'GB', 'TB']


def serialize(serializer: str, num_gates: int, nesting_depth: int) -> None:
    """"Runs a round-trip of the serializer."""
    circuit = cirq.Circuit()
    for _ in range(num_gates):
        which = np.random.choice(['expz', 'expw', 'exp11'])
        if which == 'expw':
            q1 = cirq.GridQubit(0, np.random.randint(NUM_QUBITS))
            circuit.append(
                cirq.PhasedXPowGate(
                    phase_exponent=np.random.random(), exponent=np.random.random()
                ).on(q1)
            )
        elif which == 'expz':
            q1 = cirq.GridQubit(0, np.random.randint(NUM_QUBITS))
            circuit.append(cirq.Z(q1) ** np.random.random())
        elif which == 'exp11':
            q1 = cirq.GridQubit(0, np.random.randint(NUM_QUBITS - 1))
            q2 = cirq.GridQubit(0, q1.col + 1)
            circuit.append(cirq.CZ(q1, q2) ** np.random.random())
    cs = [circuit]
    for _ in range(1, nesting_depth):
        fc = cs[-1].freeze()
        cs.append(cirq.Circuit(fc.to_op(), fc.to_op()))
    test_circuit = cs[-1]

    if serializer == _JSON:
        data = cirq.to_json(test_circuit)
        data_size = len(data)
        cirq.read_json(json_text=data)
    elif serializer == _GZIP:
        data = cirq.to_gzip(test_circuit)
        data_size = len(data)
        cirq.read_gzip(gzip_raw=data)
    return data_size


def main(
    num_gates: int,
    nesting_depth: int,
    num_repetitions: int,
    setup: str = 'from __main__ import serialize',
):
    for serializer in [_GZIP, _JSON]:
        print()
        print(f'Using serializer "{serializer}":')
        command = f'serialize(\'{serializer}\', {num_gates}, {nesting_depth})'
        time = timeit.timeit(command, setup, number=num_repetitions)
        print(f'Round-trip serializer time: {time / num_repetitions}s')
        data_size = serialize(serializer, num_gates, nesting_depth)
        suffix_idx = 0
        while data_size > 1000:
            data_size /= 1024
            suffix_idx += 1
        print(f'Serialized data size: {data_size} {SUFFIXES[suffix_idx]}.')


def parse_arguments(args):
    parser = argparse.ArgumentParser('Benchmark a serializer.')
    parser.add_argument(
        '--num_gates', default=100, type=int, help='Number of gates at the bottom nesting layer.'
    )
    parser.add_argument(
        '--nesting_depth', default=1, type=int,
        help='Depth of nested subcircuits. Total gate count will be 2^nesting_depth * num_gates.'
    )
    parser.add_argument(
        '--num_repetitions', default=10, type=int, help='Number of times to repeat serialization.'
    )
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
