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
"""Protocol for determining commutativity."""

from typing import Any, TypeVar, Union
from typing_extensions import Protocol

import numpy as np

from cirq import linalg, ops
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

    def _commutes_(self, other: Any,
                   atol: float) -> Union[None, bool, NotImplementedType]:
        r"""Determines if this object commutes with the other object.

        These matrices are the terms in the operator sum representation of
        a quantum channel. If the returned matrices are {A_0,A_1,..., A_{r-1}},
        then this describes the channel:
            \rho \rightarrow \sum_{k=0}^{r-1} A_0 \rho A_0^\dagger
        These matrices are required to satisfy the trace preserving condition
            \sum_{k=0}^{r-1} A_i^\dagger A_i = I
        where I is the identity matrix. The matrices A_i are sometimes called
        Krauss or noise operators.

        This method is used by the global `cirq.channel` method. If this method
        or the _unitary_ method is not present, or returns NotImplement,
        it is assumed that the receiving object doesn't have a channel
        (resulting in a TypeError or default result when calling `cirq.channel`
        on it). (The ability to return NotImplemented is useful when a class
        cannot know if it is a channel until runtime.)

        The order of cells in the matrices is always implicit with respect to
        the object being called. For example, for GateOperations these matrices
        must be ordered with respect to the list of qubits that the channel is
        applied to. The qubit-to-amplitude order mapping matches the
        ordering of numpy.kron(A, B), where A is a qubit earlier in the list
        than the qubit B.

        Args:
            other: The other object that may or may not commute with this
                object.
            atol: Absolute error tolerance. Some objects that commute may appear
                to not quite commute, due to rounding error from floating point
                computations. This parameter indicates an acceptable level of
                deviation from exact commutativity. The exact meaning of what
                error is being tolerated is not specified. It could be the
                maximum angle between rotation axes in the Bloch sphere, or the
                maximum trace of the absolute value of the commutator, or
                some other value convenient to the implementor of the method.
        Returns:
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

    - Either value has a `_commutes_` method that returns something besides
        `NotImplemented`. If the object's `_commutes_` method returns `False` or
        `True` then that is considered to be a definitive answer. If the
        object's method returns `None` then the commutativity is considered to
        be indeterminate. `v1._commutes_` is attempted before `v2._commutes_`.
    - Both values are matrices. The return value indicates whether these two
      matrices commute.

    If none of these techniques succeeds, it is assumed that the values do not
    commute. The order in which techniques are attempted is
    unspecified.

    Args:
        v1: One of the values to check for commutativity. Can be a cirq object
            such as an operation, or a numpy matrix.
        v2: The other value to check for commutativity. Can be a cirq object
            such as an operation, or a numpy matrix.
        default: A fallback value to return, instead of raising a ValueError, if
            there is no implemented rule for efficiently determining if the two
            values commute or not.
        atol: The minimum absolute tolerance. See np.isclose() documentation for
              details. Defaults to 1e-8 which matches np.isclose() default
              absolute tolerance.

    Returns:
        True: `v1` and `v2` commute.
        False: `v1` and `v2` don't commute.
        default: The `default` argument was specified. and there was no rule to
            efficiently determine commutativity, and

    Raises:
        TypeError: default was not set and there was no implemented rule for
            efficiently determining if the two values commute.
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
