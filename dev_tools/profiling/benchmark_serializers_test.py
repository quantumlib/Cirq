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

from dev_tools.profiling import benchmark_serializers


def test_gzip_serializer():
    for num_gates in (10, 20):
        for nesting_depth in (1, 8):
            benchmark_serializers.serialize('json_gzip', num_gates, nesting_depth)


def test_json_serializer():
    for num_gates in (10, 20):
        for nesting_depth in (1, 8):
            benchmark_serializers.serialize('json', num_gates, nesting_depth)


def test_args_have_defaults():
    kwargs = benchmark_serializers.parse_arguments([])
    for _, v in kwargs.items():
        assert v is not None


def test_main_loop():
    # Keep test from taking a long time by lowering max qubits.
    benchmark_serializers.main(
        **benchmark_serializers.parse_arguments({}),
        setup='from dev_tools.profiling.benchmark_serializers import serialize',
    )


def test_parse_args():
    args = ('--num_gates 5 --nesting_depth 8 --num_repetitions 2').split()
    kwargs = benchmark_serializers.parse_arguments(args)
    assert kwargs == {'num_gates': 5, 'nesting_depth': 8, 'num_repetitions': 2}
