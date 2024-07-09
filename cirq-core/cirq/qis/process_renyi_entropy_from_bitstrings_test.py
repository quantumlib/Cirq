# Copyright 2024 The Cirq Developers
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

from cirq.qis.process_renyi_entropy_from_bitstrings import process_entropy_from_bitstrings


@pytest.mark.parametrize('parallelize', [True, False])
def test_process_entropy_from_bitstrings(parallelize):
    bitstrings = np.array(
        [
            [[0, 1, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 1, 1]],
            [[0, 1, 1, 0], [0, 1, 1, 0], [1, 1, 0, 0], [1, 1, 0, 1]],
            [[0, 0, 1, 1], [0, 0, 0, 0], [0, 1, 1, 1], [0, 1, 1, 1]],
            [[1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 0, 0]],
            [[1, 0, 1, 1], [0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        ]
    )
    substsytem = (0, 1)
    entropy = process_entropy_from_bitstrings(bitstrings, substsytem, parallelize)
    assert entropy == 0.7


def test_process_entropy_from_bitstrings_safeguards_against_divide_by_0_error():
    bitstrings = np.array([[[0, 1, 1, 0]], [[0, 1, 1, 0]], [[0, 0, 1, 1]]])

    entropy = process_entropy_from_bitstrings(bitstrings)
    assert entropy == 0
