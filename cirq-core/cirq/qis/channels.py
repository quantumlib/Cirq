# Copyright 2021 The Cirq Developers
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
"""Tools for analyzing and manipulating quantum channels."""

import numpy as np

from cirq import protocols


def choi(operation: 'protocols.SupportsChannel') -> np.ndarray:
    """Returns the unique Choi matrix associated with a superoperator."""
    ks = protocols.channel(operation)
    d = np.prod(ks[0].shape)
    c = np.zeros((d, d), dtype=np.complex128)
    for k in ks:
        v = np.reshape(k, d)
        c += np.outer(v, v.conj())
    return c
