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

import pytest

import numpy as np

from cirq.sim import random


def test_prng_from_seed():
    assert random.prng_from_seed(None) == np.random
    assert random.prng_from_seed(np.random) == np.random

    prng = random.prng_from_seed(10)
    np.testing.assert_equal(prng.randint(0, 10, 3), [9, 4, 0])

    prng = random.prng_from_seed(np.random.RandomState(seed=4))
    np.testing.assert_equal(prng.randint(0, 10, 3), [7, 5, 1])


def test_prng_wrong_type():
    with pytest.raises(ValueError, match='float'):
        _ = random.prng_from_seed(1.0)
