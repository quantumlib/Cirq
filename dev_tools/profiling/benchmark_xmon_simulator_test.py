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

"""Tests for benchmark_xmon_simulator."""

import pytest

from absl import app
from absl.testing import flagsaver

from dev_tools.profiling import benchmark_xmon_simulator


def test_sanity_param_combos():
    for num_qubits in (4, 10):
        for num_gates in (10, 20):
            for num_prefix_qubits in (0, 2):
                for use_processes in (True, False):
                    benchmark_xmon_simulator.simulate(
                        num_qubits,
                        num_gates,
                        num_prefix_qubits,
                        use_processes)


@flagsaver.flagsaver(min_num_qubits=4, max_num_qubits=6)
def test_main():
    with pytest.raises(SystemExit):
        app.run(benchmark_xmon_simulator.main, argv=(
            'main',
            'from dev_tools.profiling.benchmark_xmon_simulator import simulate'
        ))
