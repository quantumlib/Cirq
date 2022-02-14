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

"""Tool for benchmarking serialization of large circuits.

This tool was originally introduced to enable comparison of the two JSON
serialization protocols (gzip and non-gzip):
https://github.com/quantumlib/Cirq/pull/3662

This is part of the "efficient serialization" effort:
https://github.com/quantumlib/Cirq/issues/3438

Run this benchmark with the following command (make sure to install cirq-dev):

    python3 dev_tools/profiling/benchmark_serializers.py \
        --num_gates=<int> --nesting_depth=<int> --num_repetitions=<int>

WARNING: runtime increases exponentially with nesting_depth. Values much
higher than nesting_depth=10 are not recommended.
"""

import argparse
import sys
import timeit

import numpy as np

import cirq

_JSON_GZIP = 'json_gzip'
_JSON = 'json'

NUM_QUBITS = 8

SUFFIXES = ['B', 'kB', 'MB', 'GB', 'TB']


def serialize(serializer: str, num_gates: int, nesting_depth: int) -> int:
    """Runs a round-trip of the serializer."""
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
        json_data = cirq.to_json(test_circuit)
        assert json_data is not None
        data_size = len(json_data)
        cirq.read_json(json_text=json_data)
    elif serializer == _JSON_GZIP:
        gzip_data = cirq.to_json_gzip(test_circuit)
        assert gzip_data is not None
        data_size = len(gzip_data)
        cirq.read_json_gzip(gzip_raw=gzip_data)
    return data_size


def main(
    num_gates: int,
    nesting_depth: int,
    num_repetitions: int,
    setup: str = 'from __main__ import serialize',
):
    for serializer in [_JSON_GZIP, _JSON]:
        print()
        print(f'Using serializer "{serializer}":')
        command = f'serialize(\'{serializer}\', {num_gates}, {nesting_depth})'
        time = timeit.timeit(command, setup, number=num_repetitions)
        print(f'Round-trip serializer time: {time / num_repetitions}s')
        data_size = float(serialize(serializer, num_gates, nesting_depth))
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
        '--nesting_depth',
        default=1,
        type=int,
        help='Depth of nested subcircuits. Total gate count will be 2^nesting_depth * num_gates.',
    )
    parser.add_argument(
        '--num_repetitions', default=10, type=int, help='Number of times to repeat serialization.'
    )
    return vars(parser.parse_args(args))


if __name__ == '__main__':
    main(**parse_arguments(sys.argv[1:]))
