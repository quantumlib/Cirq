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

from typing import Any

import numpy as np

from cirq import protocols
from cirq.testing import lin_alg_utils


def assert_mixture_is_consistent_with_unitary(val: Any, ignoring_global_phase: bool = False):
    """Uses `cirq.unitary` to check `cirq.mixture`'s behavior."""
    # __tracebackhide__ = True

    expected = protocols.unitary(val, None)
    if expected is None:
        # If there's no unitary, it's vacuously consistent.
        return

    has_mix = protocols.has_mixture(val)
    mix = protocols.mixture(val, None)

    # there is unitary and hence must have mixture representation.
    assert has_mix
    assert len(mix) == 1
    prob, unitary = mix[0]
    assert prob == 1

    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(unitary, expected, atol=1e-8)
    else:
        # coverage: ignore
        np.testing.assert_allclose(unitary, expected, atol=1e-8)
