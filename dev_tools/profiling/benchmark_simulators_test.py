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

"""Tests for the simulator benchmarker."""

from dev_tools.profiling import benchmark_simulators


def test_xmon_simulator():
    for num_qubits in (4, 10):
        for num_gates in (10, 20):
            for num_prefix_qubits in (0, 2):
                for use_processes in (True, False):
                    benchmark_simulators.simulate('xmon',
                                                  num_qubits,
                                                  num_gates,
                                                  num_prefix_qubits,
                                                  use_processes)


def test_unitary_simulator():
    for num_qubits in (4, 10):
        for num_gates in (10, 20):
            benchmark_simulators.simulate('unitary', num_qubits, num_gates)


def test_args_have_defaults():
    kwargs = benchmark_simulators.parse_arguments([])
    for _, v in kwargs.items():
        assert v is not None


def test_main_loop():
    # Keep test from taking a long time by lowering max qubits.
    args = '--max_num_qubits 5'.split()
    benchmark_simulators.main(
        **benchmark_simulators.parse_arguments(args),
        setup='from dev_tools.profiling.benchmark_simulators import simulate')


def test_parse_args():
    args = (
        '--sim_type unitary --min_num_qubits 5 --max_num_qubits 10 '
        '--num_gates 5 --num_repetitions 2 --num_prefix_qubits 2 '
        '--use_processes'
    ).split()
    kwargs = benchmark_simulators.parse_arguments(args)
    assert kwargs == {'sim_type': 'unitary', 'min_num_qubits': 5,
                      'max_num_qubits': 10, 'num_gates': 5,
                      'num_repetitions': 2, 'num_prefix_qubits': 2,
                      'use_processes': True}
