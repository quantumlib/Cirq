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

from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
    Optional,
    Tuple,
    List,
    Sequence,
)

import numpy as np
from typing_extensions import Protocol

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq

TDefault = TypeVar('TDefault')


class SupportsExplicitHasMixture(Protocol):
    """An object that explicitly specifies whether it has a mixture effect.

    A mixture effect is a unitary sampled from a probability distribution.
    """

    def _has_mixture_(self) -> bool:
        """Whether this value has a mixture representation.

        This method is the first thing checked by the global `cirq.has_mixture`
        method. If this method is not present, fallback strategies are used
        such as checking for `_has_unitary_`, `_mixture_`, etc.

        Returns:
            True if the value has a mixture representation, False otherwise.
        """


def has_mixture(val: Any) -> bool:
    """Determines whether the value has a unitary effect.

    Determines whether `val` has a unitary effect by attempting the following
    strategies:

    1. Try to use `val.has_unitary()`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns `True`.
            Unitary.
        Case c) Method returns `False`.
            Not unitary.

    2. Try to use `val._decompose_()`.
        Case a) Method not present or returns `NotImplemented` or `None`.
            Inconclusive.
        Case b) Method returns an OP_TREE containing only unitary operations.
            Unitary.
        Case c) Method returns an OP_TREE containing non-unitary operations.
            Not Unitary.

    3. Try to use `val._apply_unitary_(args)`.
        Case a) Method not present or returns `NotImplemented`.
            Inconclusive.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    4. Try to use `val._unitary_()`.
        Case a) Method not present or returns `NotImplemented`.
            Continue to next strategy.
        Case b) Method returns a numpy array.
            Unitary.
        Case c) Method returns `None`.
            Not unitary.

    It is assumed that, when multiple of these strategies give a conclusive
    result, that these results will all be consistent with each other. If all
    strategies are inconclusive, the value is classified as non-unitary.

    Args:
        The value that may or may not have a unitary effect.

    Returns:
        Whether or not `val` has a mixture effect.
    """
    strats = [
        _strat_has_mixture_from_has_mixture,
        _strat_has_mixture_from_decompose,
        _strat_has_mixture_from_has_unitary,
        _strat_has_mixture_from_apply_mixture,
        _strat_has_mixture_from_mixture,
    ]
    for strat in strats:
        result = strat(val)
        if result is not None:
            return result

    # If you can't tell that it's a mixture, it's not a mixture.
    return False


def _strat_has_mixture_from_has_mixture(val: Any) -> Optional[bool]:
    """Attempts to infer a value's mixture-ness via its _has_mixture_ method."""
    if hasattr(val, '_has_mixture_'):
        result = val._has_mixture_()
        if result is not NotImplemented:
            return result
    return None


def _strat_has_mixture_from_decompose(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _decompose_ method."""
    from cirq.protocols._util import _try_decompose_into_operations_and_qubits
    operations, _ = _try_decompose_into_operations_and_qubits(val)
    if operations is None:
        return None
    return all(has_mixture(op) for op in operations)


def _strat_has_mixture_from_has_unitary(val: Any) -> Optional[bool]:
    """Attempts to infer a value's mixture-ness via its unitary-ness."""
    import cirq.protocols
    return True if cirq.protocols.has_unitary(val) else None


def _strat_has_mixture_from_apply_mixture(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _apply_unitary_ method.
    """
    from cirq.protocols.apply_unitary import ApplyUnitaryArgs
    from cirq import linalg, line, ops

    method = getattr(val, '_apply_mixture_', None)
    if method is None:
        return None
    if isinstance(val, ops.Gate):
        val = val.on(*line.LineQubit.range(val.num_qubits()))
    if not isinstance(val, ops.Operation):
        return None

    n = len(val.qubits)
    state = linalg.one_hot(shape=(2,) * n, dtype=np.complex64)
    buffer = np.empty_like(state)
    result = method(ApplyUnitaryArgs(state, buffer, range(n)))
    if result is NotImplemented:
        return None
    return result is not None


def _strat_has_mixture_from_mixture(val: Any) -> Optional[bool]:
    """Attempts to infer a value's unitary-ness via its _mixture_ method."""
    getter = getattr(val, '_mixture_', None)
    if getter is None:
        return None
    result = getter()
    if result is NotImplemented:
        return None
    return result is not None
