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

import collections
from typing import TYPE_CHECKING, Callable, Union, Any, Tuple, Iterable, \
    TypeVar, List, Optional, overload

from typing_extensions import Protocol


from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


TValue = TypeVar('TValue')

TDefault = TypeVar('TDefault')

TError = TypeVar('TError', bound=Exception)

RaiseTypeErrorIfNotProvided = ([],)  # type: Any


def _value_error_describing_bad_operation(op: 'cirq.Operation') -> ValueError:
    return ValueError(
        "Operation doesn't satisfy the given `keep` "
        "but can't be decomposed: {!r}".format(op))


class SupportsDecompose(Protocol):
    """An object that can be decomposed into simpler operations.

    Returning `NotImplemented` means "not decomposable". Otherwise an operation,
    list of operations, or generally anything meeting the `OP_TREE` contract can
    be returned.

    For example, a Toffoli could be decomposed into Hadamards, CNOTs, and Ts.
    These operations are simpler because they act on fewer qubits. In general, a
    decomposition should move closer towards the basic 1-qubit and 2-qubit gates
    included by default in cirq. It is not necessary to get all the way there,
    just closer (callers will iteratively decompose if needed).

    DECOMPOSITIONS MUST NEVER BE CYCLIC. If B has A in its decomposition, A
    should not have B in its decomposition. Decomposition is often performed
    iteratively, until a fixed point or some desired criteria is met. Cyclic
    decompositions cause that kind of code to loop forever.
    """

    def _decompose_(self) -> Union[None, 'cirq.OP_TREE', NotImplementedType]:
        pass


class SupportsDecomposeWithQubits(Protocol):
    """An object that can be decomposed into operations on given qubits.

    Returning `NotImplemented` or `None` means "not decomposable". Otherwise an
    operation, list of operations, or generally anything meeting the `OP_TREE`
    contract can be returned.

    For example, a SWAP gate can be turned into three CNOTs. But in order to
    describe those CNOTs one must be able to talk about "the target qubit" and
    "the control qubit". This can only be done once the qubits-to-be-swapped are
    known.

    The main user of this protocol is `GateOperation`, which decomposes itself
    by delegating to its gate. The qubits argument is needed because gates are
    specified independently of target qubits and so must be told the relevant
    qubits. A `GateOperation` implements `SupportsDecompose` as long as its gate
    implements `SupportsDecomposeWithQubits`.
    """

    def _decompose_(self, qubits: Tuple['cirq.QubitId', ...]
                    ) -> Union[None, 'cirq.OP_TREE', NotImplementedType]:
        pass


def _default_decomposer(op: 'cirq.Operation'
                        ) -> Union[None, 'cirq.OP_TREE', NotImplementedType]:
    return decompose_once(op, default=NotImplemented)


# pylint: disable=function-redefined
@overload
def decompose(
    val: Any,
    *,
    intercepting_decomposer: Callable[['cirq.Operation'],
                                      Union[None,
                                            NotImplementedType,
                                            'cirq.OP_TREE']] = None,
    fallback_decomposer: Callable[['cirq.Operation'],
                                  Union[None,
                                        NotImplementedType,
                                        'cirq.OP_TREE']] = None,
    keep: Callable[['cirq.Operation'], bool] = None
) -> List['cirq.Operation']:
    pass


@overload
def decompose(
    val: Any,
    *,
    intercepting_decomposer: Callable[['cirq.Operation'],
                                      Union[None,
                                            NotImplementedType,
                                            'cirq.OP_TREE']] = None,
    fallback_decomposer: Callable[['cirq.Operation'],
                                  Union[None,
                                        NotImplementedType,
                                        'cirq.OP_TREE']] = None,
    keep: Callable[['cirq.Operation'], bool] = None,
    on_stuck_raise: Optional[Union[
        TError,
        Callable[['cirq.Operation'], TError]]
    ]
) -> List['cirq.Operation']:
    pass


def decompose(
    val: TValue,
    *,
    intercepting_decomposer: Callable[['cirq.Operation'],
                                      Union[None,
                                            NotImplementedType,
                                            'cirq.OP_TREE']] = None,
    fallback_decomposer: Callable[['cirq.Operation'],
                                  Union[None,
                                        NotImplementedType,
                                        'cirq.OP_TREE']] = None,
    keep: Callable[['cirq.Operation'], bool] = None,
    on_stuck_raise=_value_error_describing_bad_operation
) -> List['cirq.Operation']:
    """Recursively decomposes a value into `cirq.Operation`s meeting a criteria.

    Args:
        val: The value to decompose into operations.
        intercepting_decomposer: An optional method that is called before the
            default decomposer (the value's `_decompose_` method). If
            `intercepting_decomposer` is specified and returns a result that
            isn't `NotImplemented` or `None`, that result is used. Otherwise the
            decomposition falls back to the default decomposer.

            Note that `val` will be passed into `intercepting_decomposer`, even
            if `val` isn't a `cirq.Operation`.
        fallback_decomposer: An optional decomposition that used after the
            `intercepting_decomposer` and the default decomposer (the value's
            `_decompose_` method) both fail.
        keep: A predicate that determines if the initial operation or
            intermediate decomposed operations should be kept or else need to be
            decomposed further. If `keep` isn't specified, it defaults to "value
            can't be decomposed anymore".
        on_stuck_raise: If there is an operation that can't be decomposed and
            also can't be kept, `on_stuck_raise` is used to determine what error
            to raise. `on_stuck_raise` can either directly be an `Exception`, or
            a method that takes the problematic operation and returns an
            `Exception`. If `on_stuck_raise` is set to `None`, undecomposable
            operations are simply silently kept. `on_stuck_raise` defaults to a
            `ValueError` describing the unwanted undecomposable operation.

    Returns:
        A list of operations that the given value was decomposed into. If
        `on_stuck_raise` isn't set to None, all operations in the list will
        satisfy the predicate specified by `keep`.

    Raises:
        TypeError:
            `val` isn't a `cirq.Operation` and can't be decomposed even once.
            (So it's not possible to return a list of operations.)

        ValueError:
            Default type of error raised if there's an undecomposable operation
            that doesn't satisfy the given `keep` predicate.

        TError:
            Custom type of error raised if there's an undecomposable operation
            that doesn't satisfy the given `keep` predicate.
    """
    from cirq import ops  # HACK: Avoids circular dependencies.

    if (on_stuck_raise is not _value_error_describing_bad_operation and
            keep is None):
        raise ValueError(
            "Must specify 'keep' if specifying 'on_stuck_raise', because it's "
            "not possible to get stuck if you don't have a criteria on what's "
            "acceptable to keep.")

    decomposers = [d
                   for d in [intercepting_decomposer,
                             _default_decomposer,
                             fallback_decomposer]
                   if d]

    def decomposer(op):
        for d in decomposers:
            r = d(op)
            if r is not NotImplemented and r is not None:
                return r
        return NotImplemented

    output = []
    queue = [val]  # type: List[Any]
    while queue:
        item = queue.pop(0)

        if isinstance(item, ops.Operation) and keep is not None and keep(item):
            output.append(item)
            continue

        decomposed = decomposer(item)
        if decomposed is not NotImplemented and decomposed is not None:
            queue[:0] = ops.flatten_op_tree(decomposed)
            continue

        if isinstance(item, collections.Iterable):
            queue[:0] = ops.flatten_op_tree(item)
            continue

        if keep is not None and on_stuck_raise is not None:
            if isinstance(on_stuck_raise, Exception):
                raise on_stuck_raise
            else:
                raise on_stuck_raise(item)

        output.append(item)

    return output


@overload
def decompose_once(val: Any, **kwargs) -> List['cirq.Operation']:
    pass


@overload
def decompose_once(val: Any,
                   default: TDefault,
                   **kwargs
                   ) -> Union[TDefault, List['cirq.Operation']]:
    pass


def decompose_once(val: Any,
                   default=RaiseTypeErrorIfNotProvided,
                   **kwargs):
    """Decomposes a value into operations, if possible.

    This method decomposes the value exactly once, instead of decomposing it
    and then continuing to decomposing the decomposed operations recursively
    until some criteria is met (which is what `cirq.decompose` does).

    Args:
        val: The value to call `_decompose_` on, if possible.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented` or
            `None`. If not specified, undecomposable values cause a `TypeError`.
        kwargs: Arguments to forward into the `_decompose_` method of `val`.
            For example, this is used to tell gates what qubits they are being
            applied to.

    Returns:
        The result of `val._decompose_(**kwargs)`, if `val` has a `_decompose_`
        method and it didn't return `NotImplemented` or `None`. Otherwise
        `default` is returned, if it was specified. Otherwise an error is
        raised.

    TypeError:
        `val` didn't have a `_decompose_` method (or that method returned
        `NotImplemented` or `None`) and `default` wasn't set.
    """
    method = getattr(val, '_decompose_', None)
    decomposed = NotImplemented if method is None else method(**kwargs)

    if decomposed is not NotImplemented and decomposed is not None:
        from cirq import ops  # HACK: Avoids circular dependencies.
        return list(ops.flatten_op_tree(decomposed))

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError("object of type '{}' "
                        "has no _decompose_ method.".format(type(val)))
    raise TypeError("object of type '{}' does have a _decompose_ method, "
                    "but it returned NotImplemented or None.".format(type(val)))


@overload
def decompose_once_with_qubits(val: Any,
                               qubits: Iterable['cirq.QubitId']
                               ) -> List['cirq.Operation']:
    pass


@overload
def decompose_once_with_qubits(val: Any,
                               qubits: Iterable['cirq.QubitId'],
                               # NOTE: should be TDefault instead of Any, but
                               # mypy has false positive errors when setting
                               # default to None.
                               default: Any,
                               ) -> Union[TDefault, List['cirq.Operation']]:
    pass


def decompose_once_with_qubits(val: Any,
                               qubits: Iterable['cirq.QubitId'],
                               default=RaiseTypeErrorIfNotProvided):
    """Decomposes a value into operations on the given qubits.

    This method is used when decomposing gates, which don't know which qubits
    they are being applied to unless told. It decomposes the gate exactly once,
    instead of decomposing it and then continuing to decomposing the decomposed
    operations recursively until some criteria is met.

    Args:
        val: The value to call `._decompose_(qubits=qubits)` on, if possible.
        qubits: The value to pass into the named `qubits` parameter of
            `val._decompose_`.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented` or
            `None`. If not specified, undecomposable values cause a `TypeError`.

    Returns:
        The result of `val._decompose_(qubits=qubits)`, if `val` has a
        `_decompose_` method and it didn't return `NotImplemented` or `None`.
        Otherwise `default` is returned, if it was specified. Otherwise an error
        is raised.

    TypeError:
        `val` didn't have a `_decompose_` method (or that method returned
        `NotImplemented` or `None`) and `default` wasn't set.
    """
    return decompose_once(val, default, qubits=tuple(qubits))
# pylint: enable=function-redefined
