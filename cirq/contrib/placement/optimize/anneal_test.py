# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from unittest.mock import Mock, call

from cirq.contrib.placement.optimize import anneal


def test_accept_accepts():
    # Cost constant, should be accepted.
    assert anneal._accept(0.0, 0.0, 1.0)[0]

    # Cost improved, should be accepted.
    assert anneal._accept(0.0, -0.1, 1.0)[0]

    # Cost decreased, should be accepted if low sample.
    assert anneal._accept(0.0, 1.0, 1.0)[0]

    # Cost decreased, should be accepted if below the threshold (exp(-1.0))
    assert anneal._accept(1.0 / math.e - 1e-9, 1.0, 1.0)[0]


def test_accept_rejects():
    # Cost decreased, should be rejected if high sample.
    assert not anneal._accept(1.0 - 1e-9, 1.0, 1.0)[0]

    # Cost decreased, should be rejected if above the threshold (exp(-1.0))
    assert not anneal._accept(1.0 / math.e + 1e-9, 1.0, 1.0)[0]


def test_anneal_minimize_improves_when_better():
    assert anneal.anneal_minimize(
        'initial', lambda s: 1.0 if s == 'initial' else 0.0,
        lambda s: 'better', lambda: 1.0, 1.0, 0.5, 0.5, 1) == 'better'


def test_anneal_minimize_keeps_when_worse_and_discarded():
    assert anneal.anneal_minimize(
        'initial', lambda s: 0.0 if s == 'initial' else 1.0,
        lambda s: 'better', lambda: 0.9, 1.0, 0.5, 0.5, 1) == 'initial'


def test_anneal_minimize_raises_when_wrong_cooling_factor():
    try:
        anneal.anneal_minimize(
            'initial', lambda s: 1.0 if s == 'initial' else 0.0,
            lambda s: 'better', lambda: 1.0, 1.0, 0.5, 2.0, 1)
    except ValueError:
        return
    assert False


def test_anneal_minimize_calls_trace_func():
    trace_func = Mock()

    anneal.anneal_minimize(
        'initial', lambda s: 1.0 if s == 'initial' else 0.0,
        lambda s: 'better', lambda: 1.0, 1.0, 0.5, 0.5, 1,
        trace_func=trace_func)

    trace_func.assert_has_calls([call('initial', 1.0, 1.0, 1.0, True),
                                 call('better', 1.0, 0.0, 1.0, True)])
