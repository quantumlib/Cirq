# Copyright 2018 Google LLC
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


from typing import Any, Callable, Iterable, Tuple, Optional, TypeVar, Union

import collections

from cirq.ops import raw_types


TInternalQubit = TypeVar('TInternalQubit')
TExternalQubit = TypeVar('TExternalQubit')


class QubitOrder:
    """Defines the kronecker product order of qubits."""

    def __init__(self, explicit_func: Callable[[Iterable[raw_types.QubitId]],
                                               Tuple[raw_types.QubitId, ...]]
                 ) -> None:
        self._explicit_func = explicit_func

    DEFAULT = None  # type: QubitOrder
    """A basis that orders qubits based on their names."""

    @staticmethod
    def explicit(fixed_qubits: Iterable[raw_types.QubitId],
                 fallback: Optional['QubitOrder']=None) -> 'QubitOrder':
        """A basis that contains exactly the given qubits in the given order.

        Args:
            fixed_qubits: The qubits in basis order.
            fallback: A fallback basis to use for extra qubits not in the
                fixed_qubits list. Extra qubits will always come after the
                fixed_qubits, but will be ordered based on the fallback. If no
                fallback is specified, a ValueError is raised when extra qubits
                are specified.

        Returns:
            A Basis instance that forces the given qubits in the given order.
        """
        result = tuple(fixed_qubits)
        if len(set(result)) < len(result):
            raise ValueError(
                'Qubits appear in basis twice: {}.'.format(result))

        def func(qubits):
            remaining = set(qubits) - set(fixed_qubits)
            if not remaining:
                return result
            if not fallback:
                raise ValueError(
                    'Unexpected extra qubits: {}.'.format(remaining))
            return result + fallback.order_for(remaining)

        return QubitOrder(func)

    @staticmethod
    def sorted_by(key: Callable[[Any], Any]) -> 'QubitOrder':
        """A basis that orders qubits ascending based on a key function.

        Args:
            key: A function that takes a qubit and returns a key value. The
                basis will be ordered ascending according to these key values.


        Returns:
            A basis that orders qubits ascending based on a key function.
        """
        return QubitOrder(lambda qubits: tuple(sorted(qubits, key=key)))

    def order_for(self, qubits: Iterable[raw_types.QubitId]
                  ) -> Tuple[raw_types.QubitId, ...]:
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
    def as_qubit_order(val: 'QubitOrderOrList') -> 'QubitOrder':
        """Converts a value into a basis.

        Args:
            val: An iterable or a basis.

        Returns:
            The basis implied by the value.
        """
        if isinstance(val, collections.Iterable):
            return QubitOrder.explicit(val)
        if isinstance(val, QubitOrder):
            return val
        raise ValueError(
            "Don't know how to interpret <{}> as a Basis.".format(val))

    def map(self,
            internalize: Callable[[TExternalQubit], TInternalQubit],
            externalize: Callable[[TInternalQubit], TExternalQubit]
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


def default_sorting_key(value: Any) -> str:
    """A str method with hacks to support better lexicographic ordering.

    The output strings are not intended to be human readable.

    The returned string will have digit-runs zero-padded up to at least 8
    digits. That way, instead of 'a10' coming before 'a2', 'a000010' will come
    after 'a000002'.

    Also, the original length of each digit-run is appended after the
    zero-padded run. This is so that 'a0' continues to come before 'a00'.
    """

    text = str(value)

    was_on_digits = False
    last_transition = 0
    chunks = []

    def handle_transition_at(k):
        chunk = text[last_transition:k]
        if was_on_digits:
            chunk = chunk.rjust(8, '0') + ':' + str(len(chunk))
        chunks.append(chunk)

    for i in range(len(text)):
        on_digits = text[i].isdigit()
        if was_on_digits != on_digits:
            handle_transition_at(i)
            was_on_digits = on_digits
            last_transition = i

    handle_transition_at(len(text))
    return ''.join(chunks)


QubitOrder.DEFAULT = QubitOrder.sorted_by(default_sorting_key)

QubitOrderOrList = Union[
    QubitOrder,
    Iterable[raw_types.QubitId]]
