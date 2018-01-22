# Copyright 2018 Google LLC
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

"""Tool to benchmark the xmon_simulator."""

import timeit

import numpy as np

from absl import app
from absl import flags

from cirq.sim.google import xmon_stepper

FLAGS = flags.FLAGS

flags.DEFINE_integer('min_num_qubits', 4,
                     'Lower range (inclusive) of number of qubits that will be '
                     'benchmarked.')
flags.DEFINE_integer('max_num_qubits', 26,
                     'Upper range (inclusive) of number of qubits that will be '
                     'benchmarked.')
flags.DEFINE_integer('num_gates', 100, 'The number of gates in the benchmark.')
flags.DEFINE_integer('num_repetitions', 10,
                     'The number of times to average the benchmark over.')


def simulate(num_qubits, num_gates):
    """"Runs the xmon_simulator."""
    ops = []
    for _ in range(num_gates):
        which = np.random.choice(['z', 'xy', 'cz'])
        if which == 'xy':
            ops.append(('xy', np.random.randint(num_qubits), np.random.random(),
                        np.random.random()))
        elif which == 'z':
            ops.append(('z', np.random.randint(num_qubits), np.random.random()))
        elif which == 'cz':
            ops.append(('z', np.random.randint(num_qubits), np.random.random(),
                        np.random.random()))

    current_moment = num_qubits * [0]
    moments = [[]]

    for op in ops:
        if op[0] == 'xy' or op[0] == 'z':
            index = op[1]
            new_moment = current_moment[index]
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index] = new_moment + 1
        elif op[0] == 'cz':
            index0 = op[1]
            index1 = op[2]
            new_moment = max(index0, index1)
            if len(moments) == new_moment:
                moments.append([])
            moments[new_moment].append(op)
            current_moment[index0] = new_moment + 1
            current_moment[index1] = new_moment + 1

    with xmon_stepper.XmonSimulator(
        num_qubits=num_qubits, num_prefix_qubits=2) as s:
        for moment in moments:
            phase_map = {}
            for op in moment:
                if op[0] == 'z':
                    phase_map[(op[1],)] = op[2]
                elif op[0] == 'cz':
                    phase_map[(op[1], op[2])] = op[3]
                elif op[0] == 'xy':
                    s.simulate_xy(op[1], op[2], op[3])


def main(argv):
    del argv
    print('num_qubits,seconds per gate')
    for num_qubits in range(FLAGS.min_num_qubits, FLAGS.max_num_qubits + 1):
        time = timeit.timeit('simulate(%s, %s)' % (num_qubits, FLAGS.num_gates),
                             'from __main__ import simulate',
                             number=FLAGS.num_repetitions)
        print('%s,%s' %
              (num_qubits, time / (FLAGS.num_repetitions * FLAGS.num_gates)))


if __name__ == '__main__':
    app.run(main)
