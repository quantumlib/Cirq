# Copyright 2023 The Cirq Developers
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

from typing import Any, Callable, Hashable, Iterable, Optional, Union, overload

import attr
import cachetools
import cirq
from cirq_ft.infra.decompose_protocol import _decompose_once_considering_known_decomposition
from typing_extensions import Literal, Protocol
from cirq_ft.deprecation import deprecated_cirq_ft_class, deprecated_cirq_ft_function

_T_GATESET = cirq.Gateset(cirq.T, cirq.T**-1, unroll_circuit_op=False)


@deprecated_cirq_ft_class()
@attr.frozen
class TComplexity:
    """Dataclass storing counts of logical T-gates, Clifford gates and single qubit rotations."""

    t: int = 0
    clifford: int = 0
    rotations: int = 0

    def __add__(self, other: 'TComplexity') -> 'TComplexity':
        return TComplexity(
            self.t + other.t, self.clifford + other.clifford, self.rotations + other.rotations
        )

    def __mul__(self, other: int) -> 'TComplexity':
        return TComplexity(self.t * other, self.clifford * other, self.rotations * other)

    def __rmul__(self, other: int) -> 'TComplexity':
        return self.__mul__(other)

    def __str__(self) -> str:
        return (
            f'T-count:   {self.t:g}\n'
            f'Rotations: {self.rotations:g}\n'
            f'Cliffords: {self.clifford:g}\n'
        )


class SupportsTComplexity(Protocol):
    """An object whose TComplexity can be computed.

    An object whose TComplexity can be computed either implements the `_t_complexity_` function
    or is of a type that SupportsDecompose.
    """

    def _t_complexity_(self) -> TComplexity:
        """Returns the TComplexity."""


def _has_t_complexity(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    """Returns TComplexity of stc by calling `stc._t_complexity_()` method, if it exists."""
    estimator = getattr(stc, '_t_complexity_', None)
    if estimator is not None:
        result = estimator()
        if result is not NotImplemented:
            return result
    if isinstance(stc, cirq.Operation) and stc.gate is not None:
        return _has_t_complexity(stc.gate, fail_quietly)
    return None


def _is_clifford_or_t(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    """Attempts to infer the type of a gate/operation as one of clifford, T or Rotation."""
    if not isinstance(stc, (cirq.Gate, cirq.Operation)):
        return None

    if isinstance(stc, cirq.ClassicallyControlledOperation):
        stc = stc.without_classical_controls()

    if cirq.num_qubits(stc) <= 2 and cirq.has_stabilizer_effect(stc):
        # Clifford operation.
        return TComplexity(clifford=1)

    if stc in _T_GATESET:
        # T-gate.
        return TComplexity(t=1)  # T gate

    if cirq.num_qubits(stc) == 1 and cirq.has_unitary(stc):
        # Single qubit rotation operation.
        return TComplexity(rotations=1)
    return None


def _is_iterable(it: Any, fail_quietly: bool) -> Optional[TComplexity]:
    if not isinstance(it, Iterable):
        return None
    t = TComplexity()
    for v in it:
        r = t_complexity(v, fail_quietly=fail_quietly)
        if r is None:
            return None
        t = t + r
    return t


def _from_decomposition(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    # Decompose the object and recursively compute the complexity.
    decomposition = _decompose_once_considering_known_decomposition(stc)
    if decomposition is None:
        return None
    return _is_iterable(decomposition, fail_quietly=fail_quietly)


def _get_hash(val: Any, fail_quietly: bool = False):
    """Returns hash keys for caching a cirq.Operation and cirq.Gate.

    The hash of a cirq.Operation changes depending on its qubits, tags,
    classical controls, and other properties it has, none of these properties
    affect the TComplexity.
    For gates and gate backed operations we intend to compute the hash of the
    gate which is a property of the Gate.

    Args:
        val: object to compute its hash.

    Returns:
        - `val.gate` if `val` is a `cirq.Operation` which has an underlying `val.gate`.
        - `val` otherwise
    """
    if isinstance(val, cirq.Operation) and val.gate is not None:
        val = val.gate
    return val


def _t_complexity_from_strategies(
    stc: Any, fail_quietly: bool, strategies: Iterable[Callable[[Any, bool], Optional[TComplexity]]]
):
    ret = None
    for strategy in strategies:
        ret = strategy(stc, fail_quietly)
        if ret is not None:
            break
    return ret


@cachetools.cached(cachetools.LRUCache(128), key=_get_hash, info=True)
def _t_complexity_for_gate_or_op(
    gate_or_op: Union[cirq.Gate, cirq.Operation], fail_quietly: bool
) -> Optional[TComplexity]:
    strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition]
    return _t_complexity_from_strategies(gate_or_op, fail_quietly, strategies)


@overload
def t_complexity(stc: Any, fail_quietly: Literal[False] = False) -> TComplexity:
    ...


@overload
def t_complexity(stc: Any, fail_quietly: bool) -> Optional[TComplexity]:
    ...


@deprecated_cirq_ft_function()
def t_complexity(stc: Any, fail_quietly: bool = False) -> Optional[TComplexity]:
    """Returns the TComplexity.

    Args:
        stc: an object to compute its TComplexity.
        fail_quietly: bool whether to return None on failure or raise an error.

    Returns:
        The TComplexity of the given object or None on failure (and fail_quietly=True).

    Raises:
        TypeError: if fail_quietly=False and the methods fails to compute TComplexity.
    """
    if isinstance(stc, (cirq.Gate, cirq.Operation)) and isinstance(stc, Hashable):
        ret = _t_complexity_for_gate_or_op(stc, fail_quietly)
    else:
        strategies = [_has_t_complexity, _is_clifford_or_t, _from_decomposition, _is_iterable]
        ret = _t_complexity_from_strategies(stc, fail_quietly, strategies)

    if ret is None and not fail_quietly:
        raise TypeError("couldn't compute TComplexity of:\n" f"type: {type(stc)}\n" f"value: {stc}")
    return ret


t_complexity.cache_clear = _t_complexity_for_gate_or_op.cache_clear  # type: ignore[attr-defined]
t_complexity.cache_info = _t_complexity_for_gate_or_op.cache_info  # type: ignore[attr-defined]
t_complexity.cache = _t_complexity_for_gate_or_op.cache  # type: ignore[attr-defined]
