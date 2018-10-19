import collections
from typing import TYPE_CHECKING, Callable, Union, Any, Tuple, Iterable, \
    TypeVar, List, Optional

from typing_extensions import Protocol


from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


TDefault = TypeVar('TDefault')

TError = TypeVar('TError', Exception)

RaiseTypeErrorIfNotProvided = ([],)


def _value_error_describing_bad_operation(op: 'cirq.Operation') -> ValueError:
    return ValueError(
        "Operation doesn't satisfy the given `keep` "
        "but can't be decomposed: {!r}".format(op))


class SupportsDecompose(Protocol):
    """An object that may be describable by a unitary matrix."""

    def _decompose_(self) -> Union['cirq.OP_TREE', NotImplementedType]:
        return NotImplemented


class SupportsDecomposeWithQubits(Protocol):
    """A gate that may be describable by a unitary matrix."""

    def _decompose_(self, qubits: Tuple['cirq.QubitId', ...]
                    ) -> Union['cirq.OP_TREE', NotImplementedType]:
        return NotImplemented


def _default_decomposer(op: 'cirq.Operation'
                        ) -> Union['cirq.OP_TREE', NotImplementedType]:
    method = getattr(op, '_decompose_', None)
    return NotImplemented if op is None else method()


def _intercept_decompose(
        op: 'cirq.Operation',
        decomposers: List[Callable[['cirq.Operation'], 'cirq.OP_TREE']]
) -> 'cirq.OP_TREE':
    for d in decomposers:
        r = d(op)
        if r is not NotImplemented:
            return r
    return NotImplemented


def decompose(
    val: Any,
    *,
    intercepting_decomposer: Callable[['cirq.Operation'],
                                      'cirq.OP_TREE'] = None,
    decomposer: Callable[['cirq.Operation'],
                         'cirq.OP_TREE'] = _default_decomposer,
    keep: Callable[['cirq.Operation'], bool] = None,
    on_stuck_raise: Optional[Union[
        TError,
        Callable[['cirq.Operation'], TError]]
    ] = _value_error_describing_bad_operation
) -> List['cirq.Operation']:
    """Recursively decomposes a value into `cirq.Operation`s.

    Args:
        val: The value to decompose into operations.
        intercepting_decomposer: An optional method that is called before the
            main decomposer when doing intermediate decompositions. If
            `intercepting_decomposer` isn't specified, or returns
            `NotImplemented`, the decomposition falls back to the main
            `decomposer`. Otherwise the result of `intercepting_decomposer`
            takes priority and `decomposer` isn't even called.
        decomposer: Decomposes operations into more basic operations, or returns
            `NotImplemented` to indicate that an operation cannot be decomposed
            anymore. Defaults to a decomposer that calls operations'
            `_decompose_` method.
        keep: A predicate that determines if the initial operation or
            intermediate decomposed operations should be kept or else need to be
            decomposed further. If `keep` isn't specified, operations are simply
            decomposed until they can't be decomposed anymore.
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

    if on_stuck_raise is not None and keep is None:
        raise ValueError(
            "Must specify 'keep' if specifying 'on_stuck_raise', because it's "
            "not possible to get stuck if you don't have a criteria on what's "
            "acceptable to keep.")

    if intercepting_decomposer is None:
        actual_decomposer = decomposer
    else:
        actual_decomposer = lambda op: _intercept_decompose(
            op, [intercepting_decomposer, decomposer])

    output = []
    queue = []  # type: List['cirq.Operation']
    if not isinstance(val, ops.Operation):
        queue.extend(decompose_once(val))
    else:
        queue.append(val)

    while queue:
        item = queue.pop(0)

        if keep is not None and keep(item):
            output.append(item)
            continue

        decomposed = actual_decomposer(item)
        if decomposed is not NotImplemented:
            queue.extend(ops.flatten_op_tree(decomposed))
            continue

        if isinstance(item, collections.Iterable):
            queue.extend(ops.flatten_op_tree(item))
            continue

        if on_stuck_raise is not None:
            if isinstance(on_stuck_raise, Exception):
                raise on_stuck_raise
            else:
                raise on_stuck_raise(item)

        output.append(item)

    return output


def decompose_once(val: Any,
                   default: TDefault=RaiseTypeErrorIfNotProvided,
                   **kwargs
                   ) -> Union[TDefault, List['cirq.Operation']]:
    """Decomposes a value into operations, if possible.

    Args:
        val: The value to call `_decompose_` on, if possible.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented`.
            If not specified, undecomposable values cause a `TypeError`.
        kwargs: Arguments to forward into the `_decompose_` method of `val`.
            For example, this is used to tell gates what qubits they are being
            applied to.

    Returns:
        The result of `val._decompose_(**kwargs)`, if `val` has a `_decompose_`
        method and it didn't return `NotImplemented`. Otherwise `default` is
        returned, if it was specified. Otherwise an error is raised.

    TypeError:
        `val` didn't have a `_decompose_` method (or that method returned
        `NotImplemented`) and `default` wasn't set.
    """
    from cirq import ops  # HACK: Avoids circular dependencies.

    method = getattr(val, '_decompose_', None)
    decomposed = NotImplemented if method is None else method(**kwargs)

    if decomposed is not NotImplemented:
        return list(ops.flatten_op_tree(decomposed))

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError("object of type '{}' "
                        "has no _decompose_ method.".format(type(val)))
    raise TypeError("object of type '{}' does have a _decompose_ method, "
                    "but it returned NotImplemented.".format(type(val)))


def decompose_once_with_qubits(
    val: Any,
    qubits: Iterable['cirq.QubitId'],
    default: TDefault = ([],)
) -> 'cirq.OP_TREE':
    """Decomposes a value into operations on the given qubits.

    This method is used when decomposing gates, which don't know which qubits
    they are being applied to unless told.

    Args:
        val: The value to call `._decompose_(qubits=qubits)` on, if possible.
        qubits: The value to pass into the named `qubits` parameter of
            `val._decompose_`.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented`.
            If not specified, undecomposable values cause a `TypeError`.

    Returns:
        The result of `val._decompose_(qubits=qubits)`, if `val` has a
        `_decompose_` method and it didn't return `NotImplemented`. Otherwise
        `default` is returned, if it was specified. Otherwise an error is
        raised.

    TypeError:
        `val` didn't have a `_decompose_` method (or that method returned
        `NotImplemented`) and `default` wasn't set.
    """
    return decompose_once(val, default, qubits=qubits)
