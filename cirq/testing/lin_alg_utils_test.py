# Copyright 2017 Google LLC
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

from cirq.testing.lin_alg_utils import random_unitary
from cirq.linalg import is_unitary


def test_random_unitary():
    u1 = random_unitary(2)
    u2 = random_unitary(2)
    assert is_unitary(u1)
    assert is_unitary(u2)
    assert not np.allclose(u1, u2)
