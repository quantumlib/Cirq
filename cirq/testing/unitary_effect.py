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

from typing import Any, Optional, Callable

import numpy as np

from cirq import protocols


def assert_unitary_effect_is(val: Any,
                             expected_effect: Optional[np.ndarray],
                             atol: float=1e-8) -> None:
    """Determines if the value has the given unitary effect (or lack thereof).

    This method also checks that the value is implementing the
    SupportsUnitaryEffect correctly (i.e. that the various methods that are
    present are consistent with each other, and behave well with the global
    helper methods such as cirq.unitary_effect).

    Args:
        val: The value that should have a specified unitary effect.
        expected_effect: The unitary effect the value should have, or None if
            the value should not have a unitary effect.
        atol: Absolute error tolerance when comparing effects.

    Raises:
        AssertionError: The expected effect doesn't match the value's actual
            effect, or the value has made some mistake in implementing the
            SupportsUnitaryEffect protocol.
    """

    has_method = getattr(val, '_has_unitary_effect_', None)
    maybe_method = getattr(val, '_maybe_unitary_effect_', None)
    get_method = getattr(val, '_unitary_effect_', None)

    if has_method is not None:
        assert has_method() == (expected_effect is not None)

    if maybe_method is not None:
        _assert_both_none_or_close(maybe_method(),
                                   expected_effect,
                                   atol=atol)

    if get_method is not None:
        _assert_both_none_or_close(_trap_get(get_method),
                                   expected_effect,
                                   atol=atol)

    if expected_effect is not None:
        assert maybe_method or get_method

    # Usage via the global methods.
    assert protocols.has_unitary_effect(val) == (expected_effect is not None)
    _assert_both_none_or_close(protocols.maybe_unitary_effect(val),
                               expected_effect,
                               atol=atol)
    _assert_both_none_or_close(_trap_get(lambda: protocols.unitary_effect(val)),
                               expected_effect,
                               atol=atol)


def _assert_both_none_or_close(a: Optional[np.ndarray],
                               b: Optional[np.ndarray],
                               atol: float):
    assert (a is None) == (b is None)
    if a is not None and b is not None:
        np.testing.assert_allclose(a, b, atol=atol)


def _trap_get(func: Callable[[], np.ndarray]) -> Optional[np.ndarray]:
    try:
        result = func()
        assert result is not None
        return result
    except ValueError:
        return None
