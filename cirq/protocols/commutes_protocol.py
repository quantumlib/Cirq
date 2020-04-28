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
"""Protocol for determining commutativity."""

from typing import Any, TypeVar, Union

import numpy as np

from typing_extensions import Protocol

from cirq import linalg, ops
from cirq._doc import document
from cirq.protocols import qid_shape_protocol, unitary_protocol
from cirq.type_workarounds import NotImplementedType

# This is a special indicator value used by the unitary method to determine
# whether or not the caller provided a 'default' argument.
# It is checked for using `is`, so it won't have a false positive if the user
# provides a different np.array([]) value.
RaiseTypeErrorIfNotProvided = np.array([])

TDefault = TypeVar('TDefault')


class SupportsCommutes(Protocol):
    """An object that can determine commutation relationships vs others."""

    @document
    def _commutes_(self, other: Any,
                   atol: float) -> Union[None, bool, NotImplementedType]:
        r"""Determines if this object commutes with the other object.

        Can return None to indicate the commutation relationship is
        indeterminate (e.g. incompatible matrix sizes). Can return
        NotImplemented to indicate to the caller that they should try some other
        way of determining the commutation relationship.

        Args:
            other: The other object that may or may not commute with the
                receiving object.
            atol: Absolute error tolerance. Some objects that commute may appear
                to not quite commute, due to rounding error from floating point
                computations. This parameter indicates an acceptable level of
                deviation from exact commutativity. The exact meaning of what
                error is being tolerated is not specified. It could be the
                maximum angle between rotation axes in the Bloch sphere, or the
                maximum trace of the absolute value of the commutator, or
                some other value convenient to the implementor of the method.

        Returns:
            Whether or not the values commute.

            True: `self` commutes with `other` within absolute tolerance `atol`.

            False: `self` does not commute with `other`.

            None: There is not a well defined commutation result. For example,
            whether or not parameterized operations will commute may depend
            on the parameter values and so is indeterminate.

            NotImplemented: Unable to determine anything about commutativity.
            Consider falling back to other strategies, such as asking
            `other` if it commutes with `self` or computing the unitary
            matrices of both values.
        """


def commutes(v1: Any,
             v2: Any,
             *,
             atol: Union[int, float] = 1e-8,
             default: TDefault = RaiseTypeErrorIfNotProvided
            ) -> Union[bool, TDefault]:
    """Determines whether two values commute.

    This is determined by any one of the following techniques:

    - Either value has a `_commutes_` method that returns 'True', 'False', or
        'None' (meaning indeterminate). If both methods either don't exist or
        return `NotImplemented` then another strategy is tried. `v1._commutes_`
        is tried before `v2._commutes_`.
    - Both values are matrices. The return value is determined by checking if
        v1 @ v2 - v2 @ v1 is sufficiently close to zero.
    - Both values are `cirq.Operation` instances. If the operations apply to
        disjoint qubit sets then they commute. Otherwise, if they have unitary
        matrices, those matrices are checked for commutativity (while accounting
        for the fact that the operations may have different qubit orders or only
        partially overlap).

    If none of these techniques succeeds, the commutativity is assumed to be
    indeterminate.

    Args:
        v1: One of the values to check for commutativity. Can be a cirq object
            such as an operation, or a numpy matrix.
        v2: The other value to check for commutativity. Can be a cirq object
            such as an operation, or a numpy matrix.
        default: A fallback value to return, instead of raising a ValueError, if
            it is indeterminate whether or not the two values commute.
        atol: Absolute error tolerance. If all entries in v1@v2 - v2@v1 have a
            magnitude less than this tolerance, v1 and v2 can be reported as
            commuting. Defaults to 1e-8.

    Returns:
        True: `v1` and `v2` commute (or approximately commute).
        False: `v1` and `v2` don't commute.
        default: The commutativity of `v1` and `v2` is indeterminate, or could
        not be determined, and the `default` argument was specified.

    Raises:
        TypeError: The commutativity of `v1` and `v2` is indeterminate, or could
        not be determined, and the `default` argument was not specified.
    """
    atol = float(atol)

    strats = [
        _strat_commutes_from_commutes,
        _strat_commutes_from_matrix,
        _strat_commutes_from_operation,
    ]
    for strat in strats:
        result = strat(v1, v2, atol=atol)
        if result is None:
            break
        if result is not NotImplemented:
            return result
    if default is not RaiseTypeErrorIfNotProvided:
        return default
    raise TypeError(
        f"Failed to determine whether or not "
        f"{v1!r} commutes with {v2!r}. "
        f"The result may be indeterminate, or there may be no strategy "
        f"implemented to handle this case.\n"
        f"If you want a default result in this case, specify a `default=` "
        f"argument or use `cirq.definitely_commutes`.")


def definitely_commutes(v1: Any, v2: Any, *,
                        atol: Union[int, float] = 1e-8) -> bool:
    """Determines whether two values definitely commute.

    Returns:
        True: The two values definitely commute.
        False: The two values may or may not commute.
    """
    return commutes(v1, v2, atol=atol, default=False)


def _strat_commutes_from_commutes(v1: Any,
                                  v2: Any,
                                  *,
                                  atol: Union[int, float] = 1e-8
                                 ) -> Union[bool, NotImplementedType, None]:
    """Attempts to determine commutativity via the objects' _commutes_
    method."""

    for a, b in [(v1, v2), (v2, v1)]:
        getter = getattr(a, '_commutes_', None)
        if getter is None:
            continue
        val = getter(b, atol=atol)
        if val is not NotImplemented:
            return val
    return NotImplemented


def _strat_commutes_from_matrix(
        v1: Any,
        v2: Any,
        *,
        atol: float,
) -> Union[bool, NotImplementedType, None]:
    """Attempts to determine commutativity of matrices."""
    if not isinstance(v1, np.ndarray) or not isinstance(v2, np.ndarray):
        return NotImplemented
    if v1.shape != v2.shape:
        return None
    return linalg.matrix_commutes(v1, v2, atol=atol)


def _strat_commutes_from_operation(
        v1: Any,
        v2: Any,
        *,
        atol: float,
) -> Union[bool, NotImplementedType, None]:
    if not isinstance(v1, ops.Operation) or not isinstance(v2, ops.Operation):
        return NotImplemented

    if set(v1.qubits).isdisjoint(v2.qubits):
        return True

    from cirq import circuits
    circuit12 = circuits.Circuit(v1, v2)
    circuit21 = circuits.Circuit(v2, v1)

    # Don't create gigantic matrices.
    if np.product(qid_shape_protocol.qid_shape(circuit12)) > 2**10:
        return NotImplemented  # coverage: ignore

    m12 = unitary_protocol.unitary(circuit12, default=None)
    m21 = unitary_protocol.unitary(circuit21, default=None)
    if m12 is None:
        return NotImplemented
    return np.allclose(m12, m21, atol=atol)
