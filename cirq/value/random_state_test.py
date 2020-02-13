# Copyright 2019 The Cirq Developers
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

import numpy as np

import cirq


def test_parse_random_state():
    global_state = np.random.get_state()

    def rand(prng):
        np.random.set_state(global_state)
        return prng.rand()

    prngs = [
        np.random,
        cirq.value.parse_random_state(np.random),
        cirq.value.parse_random_state(None)
    ]
    vals = [rand(prng) for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)

    seed = np.random.randint(2**31)
    prngs = [
        np.random.RandomState(seed),
        cirq.value.parse_random_state(np.random.RandomState(seed)),
        cirq.value.parse_random_state(seed)
    ]
    vals = [prng.rand() for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)
