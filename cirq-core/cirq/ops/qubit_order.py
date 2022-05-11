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


from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, TYPE_CHECKING

from cirq.ops import raw_types

if TYPE_CHECKING:
    from cirq.ops import qubit_order_or_list


TInternalQubit = TypeVar('TInternalQubit')
TExternalQubit = TypeVar('TExternalQubit')


class QubitOrder:
    """Defines the kronecker product order of qubits."""

    def __init__(
        self, explicit_func: Callable[[Iterable[raw_types.Qid]], Tuple[raw_types.Qid, ...]]
    ) -> None:
        self._explicit_func = explicit_func

    DEFAULT: 'QubitOrder'
    """A basis that orders qubits in the same way that calling `sorted` does.

    Specifically, qubits are ordered first by their type name and then by
    whatever comparison value qubits of a given type provide (e.g. for LineQubit
    it is the x coordinate of the qubit).
    """

    @staticmethod
    def explicit(
        fixed_qubits: Iterable[raw_types.Qid], fallback: Optional['QubitOrder'] = None
    ) -> 'QubitOrder':
        """A basis that contains exactly the given qubits in the given order.

        Args:
            fixed_qubits: The qubits in basis order.
            fallback: A fallback order to use for extra qubits not in the
                fixed_qubits list. Extra qubits will always come after the
                fixed_qubits, but will be ordered based on the fallback. If no
                fallback is specified, a ValueError is raised when extra qubits
                are specified.

        Returns:
            A Basis instance that forces the given qubits in the given order.

        Raises:
            ValueError: If a qubit appears twice in `fixed_qubits`, or there were is no fallback
                specified but there are extra qubits.
        """
        result = tuple(fixed_qubits)
        if len(set(result)) < len(result):
            raise ValueError(f'Qubits appear in fixed_order twice: {result}.')

        def func(qubits):
            remaining = set(qubits) - set(result)
            if not remaining:
                return result
            if not fallback:
                raise ValueError(f'Unexpected extra qubits: {remaining}.')
            return result + fallback.order_for(remaining)

        return QubitOrder(func)

    @staticmethod
    def sorted_by(key: Callable[[raw_types.Qid], Any]) -> 'QubitOrder':
        """A basis that orders qubits ascending based on a key function.

        Args:
            key: A function that takes a qubit and returns a key value. The
                basis will be ordered ascending according to these key values.

        Returns:
            A basis that orders qubits ascending based on a key function.
        """
        return QubitOrder(lambda qubits: tuple(sorted(qubits, key=key)))

    def order_for(self, qubits: Iterable[raw_types.Qid]) -> Tuple[raw_types.Qid, ...]:
        """Returns a qubit tuple ordered corresponding to the basis.

        Args:
            qubits: Qubits that should be included in the basis. (Additional
                qubits may be added into the output by the basis.)

        Returns:
            A tuple of qubits in the same order that their single-qubit
            matrices would be passed into `np.kron` when producing a matrix for
            the entire system.
        """
        return self._explicit_func(qubits)

    @staticmethod
    def as_qubit_order(val: 'qubit_order_or_list.QubitOrderOrList') -> 'QubitOrder':
        """Converts a value into a basis.

        Args:
            val: An iterable or a basis.

        Returns:
            The basis implied by the value.

        Raises:
            ValueError: If `val` is not an iterable or a `QubitOrder`.
        """
        if isinstance(val, Iterable):
            return QubitOrder.explicit(val)
        if isinstance(val, QubitOrder):
            return val
        raise ValueError(f"Don't know how to interpret <{val}> as a Basis.")

    def map(
        self,
        internalize: Callable[[TExternalQubit], TInternalQubit],
        externalize: Callable[[TInternalQubit], TExternalQubit],
    ) -> 'QubitOrder':
        """Transforms the Basis so that it applies to wrapped qubits.

        Args:
            externalize: Converts an internal qubit understood by the underlying
                basis into an external qubit understood by the caller.
            internalize: Converts an external qubit understood by the caller
                into an internal qubit understood by the underlying basis.

        Returns:
            A basis that transforms qubits understood by the caller into qubits
            understood by an underlying basis, uses that to order the qubits,
            then wraps the ordered qubits back up for the caller.
        """

        def func(qubits):
            unwrapped_qubits = [internalize(q) for q in qubits]
            unwrapped_result = self.order_for(unwrapped_qubits)
            return tuple(externalize(q) for q in unwrapped_result)

        return QubitOrder(func)


QubitOrder.DEFAULT = QubitOrder.sorted_by(lambda v: v)
