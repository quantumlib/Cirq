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


def assert_kraus_is_consistent_with_unitary(val: Any, ignoring_global_phase: bool = False):
    """Uses `cirq.unitary` to check `val.kraus`'s behavior."""
    # pylint: disable=unused-variable
    # __tracebackhide__ = True
    # pylint: enable=unused-variable

    expected = protocols.unitary(val, None)
    if expected is None:
        # If there's no unitary, it's vacuously consistent.
        return

    has_krs = protocols.kraus_protocol.has_kraus(val)
    krs = protocols.kraus_protocol.kraus(val, None)

    # there is unitary and hence must have kraus operator
    assert has_krs
    actual = krs[0]

    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-8)
    else:
        # coverage: ignore
        np.testing.assert_allclose(actual, expected, atol=1e-8)


def assert_kraus_is_consistent_with_mixture(val: Any, ignoring_global_phase: bool = False):
    """Uses `cirq.mixture` to check `cirq.kraus`'s behavior."""
    # pylint: disable=unused-variable
    # __tracebackhide__ = True
    # pylint: enable=unused-variable

    expected = protocols.mixture(val, None)
    if expected is None:
        # If there's no mixture, it's vacuously consistent.
        return

    has_krs = protocols.kraus_protocol.has_kraus(val)
    krs = protocols.kraus_protocol.kraus(val, None)

    # there is mixture and hence must have kraus operator
    assert has_krs
    actual = krs
    expected = np.array([np.sqrt(p) * x for p, x in expected])

    if ignoring_global_phase:
        lin_alg_utils.assert_allclose_up_to_global_phase(actual, expected, atol=1e-8)
    else:
        # coverage: ignore
        np.testing.assert_allclose(actual, expected, atol=1e-8)
