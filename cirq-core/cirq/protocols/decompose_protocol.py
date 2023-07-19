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
import itertools
import dataclasses
import inspect
from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    overload,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import runtime_checkable

from typing_extensions import Protocol

from cirq import devices, ops
from cirq._doc import doc_private
from cirq.protocols import qid_shape_protocol
from cirq.type_workarounds import NotImplementedType

if TYPE_CHECKING:
    import cirq

TDefault = TypeVar('TDefault')

TError = TypeVar('TError', bound=Exception)

RaiseTypeErrorIfNotProvided: Any = ([],)

DecomposeResult = Union[None, NotImplementedType, 'cirq.OP_TREE']

_CONTEXT_COUNTER = itertools.count()  # Use _reset_context_counter() to reset the counter.


@runtime_checkable
class OpDecomposerWithContext(Protocol):
    def __call__(
        self, __op: 'cirq.Operation', *, context: Optional['cirq.DecompositionContext'] = None
    ) -> DecomposeResult:
        ...


OpDecomposer = Union[Callable[['cirq.Operation'], DecomposeResult], OpDecomposerWithContext]

DECOMPOSE_TARGET_GATESET = ops.Gateset(
    ops.XPowGate,
    ops.YPowGate,
    ops.ZPowGate,
    ops.CZPowGate,
    ops.MeasurementGate,
    ops.GlobalPhaseGate,
)


def _value_error_describing_bad_operation(op: 'cirq.Operation') -> ValueError:
    return ValueError(f"Operation doesn't satisfy the given `keep` but can't be decomposed: {op!r}")


@dataclasses.dataclass(frozen=True)
class DecompositionContext:
    """Stores common configurable options for decomposing composite gates into simpler operations.

    Args:
        qubit_manager: A `cirq.QubitManager` instance to allocate clean / dirty ancilla qubits as
            part of the decompose protocol.
    """

    qubit_manager: 'cirq.QubitManager'


class SupportsDecompose(Protocol):
    """An object that can be decomposed into simpler operations.

    All decomposition methods should ultimately terminate on basic 1-qubit and
    2-qubit gates included by default in Cirq. Cirq does not make any guarantees
    about what the final gate set is. Currently, decompositions within Cirq
    happen to converge towards the X, Y, Z, CZ, PhasedX, specified-matrix gates,
    and others. This set will vary from release to release. Because of this
    variability, it is important for consumers of decomposition to look for
    generic properties of gates, such as "two qubit gate with a unitary matrix",
    instead of specific gate types such as CZ gates (though a consumer is
    of course free to handle CZ gates in a special way, and consumers can
    give an `intercepting_decomposer` to `cirq.decompose` that attempts to
    target a specific gate set).

    For example, `cirq.TOFFOLI` has a `_decompose_` method that returns a pair
    of Hadamard gates surrounding a `cirq.CCZ`. Although `cirq.CCZ` is not a
    1-qubit or 2-qubit operation, it specifies its own `_decompose_` method
    that only returns 1-qubit or 2-qubit operations. This means that iteratively
    decomposing `cirq.TOFFOLI` terminates in 1-qubit and 2-qubit operations, and
    so almost all decomposition-aware code will be able to handle `cirq.TOFFOLI`
    instances.

    Callers are responsible for iteratively decomposing until they are given
    operations that they understand. The `cirq.decompose` method is a simple way
    to do this, because it has logic to recursively decompose until a given
    `keep` predicate is satisfied.

    Code implementing `_decompose_` MUST NOT create cycles, such as a gate A
    decomposes into a gate B which decomposes back into gate A. This will result
    in infinite loops when calling `cirq.decompose`.

    It is permitted (though not recommended) for the chain of decompositions
    resulting from an operation to hit a dead end before reaching 1-qubit or
    2-qubit operations. When this happens, `cirq.decompose` will raise
    a `TypeError` by default, but can be configured to ignore the issue or
    raise a caller-provided error.
    """

    @doc_private
    def _decompose_(self) -> DecomposeResult:
        pass

    def _decompose_with_context_(
        self, *, context: Optional[DecompositionContext] = None
    ) -> DecomposeResult:
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

    def _decompose_(self, qubits: Tuple['cirq.Qid', ...]) -> DecomposeResult:
        pass

    def _decompose_with_context_(
        self, qubits: Tuple['cirq.Qid', ...], *, context: Optional[DecompositionContext] = None
    ) -> DecomposeResult:
        pass


def _try_op_decomposer(
    val: Any, decomposer: Optional[OpDecomposer], *, context: Optional[DecompositionContext] = None
) -> DecomposeResult:
    if decomposer is None or not isinstance(val, ops.Operation):
        return None
    if 'context' in inspect.signature(decomposer).parameters:
        assert isinstance(decomposer, OpDecomposerWithContext)
        return decomposer(val, context=context)
    else:
        return decomposer(val)


@dataclasses.dataclass(frozen=True)
class _DecomposeArgs:
    context: Optional[DecompositionContext]
    intercepting_decomposer: Optional[OpDecomposer]
    fallback_decomposer: Optional[OpDecomposer]
    keep: Optional[Callable[['cirq.Operation'], bool]]
    on_stuck_raise: Union[None, Exception, Callable[['cirq.Operation'], Optional[Exception]]]
    preserve_structure: bool


def _decompose_dfs(item: Any, args: _DecomposeArgs) -> Iterator['cirq.Operation']:
    from cirq.circuits import CircuitOperation, FrozenCircuit

    if isinstance(item, ops.Operation):
        item_untagged = item.untagged
        if args.preserve_structure and isinstance(item_untagged, CircuitOperation):
            new_fc = FrozenCircuit(_decompose_dfs(item_untagged.circuit, args))
            yield item_untagged.replace(circuit=new_fc).with_tags(*item.tags)
            return
        if args.keep is not None and args.keep(item):
            yield item
            return

    decomposed = _try_op_decomposer(item, args.intercepting_decomposer, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        decomposed = decompose_once(item, default=None, flatten=False, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        decomposed = _try_op_decomposer(item, args.fallback_decomposer, context=args.context)

    if decomposed is NotImplemented or decomposed is None:
        if not isinstance(item, ops.Operation) and isinstance(item, Iterable):
            decomposed = item

    if decomposed is NotImplemented or decomposed is None:
        if args.keep is not None and args.on_stuck_raise is not None:
            if isinstance(args.on_stuck_raise, Exception):
                raise args.on_stuck_raise
            elif callable(args.on_stuck_raise):
                error = args.on_stuck_raise(item)
                if error is not None:
                    raise error
        yield item
    else:
        for val in ops.flatten_to_ops(decomposed):
            yield from _decompose_dfs(val, args)


def decompose(
    val: Any,
    *,
    intercepting_decomposer: Optional[OpDecomposer] = None,
    fallback_decomposer: Optional[OpDecomposer] = None,
    keep: Optional[Callable[['cirq.Operation'], bool]] = None,
    on_stuck_raise: Union[
        None, Exception, Callable[['cirq.Operation'], Optional[Exception]]
    ] = _value_error_describing_bad_operation,
    preserve_structure: bool = False,
    context: Optional[DecompositionContext] = None,
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
            `Exception`. If `on_stuck_raise` is set to `None` or a method that
            returns `None`, non-decomposable operations are simply silently
            kept. `on_stuck_raise` defaults to a `ValueError` describing the
            unwanted non-decomposable operation.
        preserve_structure: Prevents subcircuits (i.e. `CircuitOperation`s)
            from being decomposed, but decomposes their contents. If this is
            True, `intercepting_decomposer` cannot be specified.
        context: Decomposition context specifying common configurable options for
            controlling the behavior of decompose.

    Returns:
        A list of operations that the given value was decomposed into. If
        `on_stuck_raise` isn't set to None, all operations in the list will
        satisfy the predicate specified by `keep`.

    Raises:
        TypeError:
            `val` isn't a `cirq.Operation` and can't be decomposed even once.
            (So it's not possible to return a list of operations.)

        ValueError:
            Default type of error raised if there's an non-decomposable
            operation that doesn't satisfy the given `keep` predicate.

        TError:
            Custom type of error raised if there's an non-decomposable operation
            that doesn't satisfy the given `keep` predicate.
    """

    if (
        on_stuck_raise is not _value_error_describing_bad_operation
        and on_stuck_raise is not None
        and keep is None
    ):
        raise ValueError(
            "Must specify 'keep' if specifying 'on_stuck_raise', because it's "
            "not possible to get stuck if you don't have a criteria on what's "
            "acceptable to keep."
        )

    if context is None:
        context = DecompositionContext(ops.SimpleQubitManager(prefix='_decompose_protocol'))
    args = _DecomposeArgs(
        context=context,
        intercepting_decomposer=intercepting_decomposer,
        fallback_decomposer=fallback_decomposer,
        keep=keep,
        on_stuck_raise=on_stuck_raise,
        preserve_structure=preserve_structure,
    )
    return [*_decompose_dfs(val, args)]


# pylint: disable=function-redefined


@overload
def decompose_once(val: Any, **kwargs) -> List['cirq.Operation']:
    pass


@overload
def decompose_once(
    val: Any, default: TDefault, *args, flatten: bool = True, **kwargs
) -> Union[TDefault, List['cirq.Operation']]:
    pass


def decompose_once(
    val: Any,
    default=RaiseTypeErrorIfNotProvided,
    *args,
    flatten: bool = True,
    context: Optional[DecompositionContext] = None,
    **kwargs,
):
    """Decomposes a value into operations, if possible.

    This method decomposes the value exactly once, instead of decomposing it
    and then continuing to decomposing the decomposed operations recursively
    until some criteria is met (which is what `cirq.decompose` does).

    Args:
        val: The value to call `_decompose_` on, if possible.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented` or
            `None`. If not specified, non-decomposable values cause a
            `TypeError`.
        *args: Positional arguments to forward into the `_decompose_` method of
            `val`.  For example, this is used to tell gates what qubits they are
            being applied to.
        flatten: If True, the returned OP-TREE will be flattened to a list of operations.
        context: Decomposition context specifying common configurable options for
            controlling the behavior of decompose.
        **kwargs: Keyword arguments to forward into the `_decompose_` method of
            `val`.

    Returns:
        The result of `val._decompose_(*args, **kwargs)`, if `val` has a
        `_decompose_` method and it didn't return `NotImplemented` or `None`.
        Otherwise `default` is returned, if it was specified. Otherwise an error
        is raised.

    Raises:
        TypeError: `val` didn't have a `_decompose_` method (or that method returned
            `NotImplemented` or `None`) and `default` wasn't set.
    """
    if context is None:
        context = DecompositionContext(
            ops.SimpleQubitManager(prefix=f'_decompose_protocol_{next(_CONTEXT_COUNTER)}')
        )

    method = getattr(val, '_decompose_with_context_', None)
    decomposed = NotImplemented if method is None else method(*args, **kwargs, context=context)
    if decomposed is NotImplemented or None:
        method = getattr(val, '_decompose_', None)
        decomposed = NotImplemented if method is None else method(*args, **kwargs)

    if decomposed is not NotImplemented and decomposed is not None:
        return list(ops.flatten_to_ops(decomposed)) if flatten else decomposed

    if default is not RaiseTypeErrorIfNotProvided:
        return default
    if method is None:
        raise TypeError(
            f"object of type '{type(val)}' has no _decompose_with_context_ or "
            f"_decompose_ method."
        )
    raise TypeError(
        f"object of type {type(val)} does have a _decompose_ method, "
        "but it returned NotImplemented or None."
    )


@overload
def decompose_once_with_qubits(
    val: Any,
    qubits: Iterable['cirq.Qid'],
    *,
    flatten: bool = True,
    context: Optional['DecompositionContext'] = None,
) -> List['cirq.Operation']:
    pass


@overload
def decompose_once_with_qubits(
    val: Any,
    qubits: Iterable['cirq.Qid'],
    default: Optional[TDefault],
    *,
    flatten: bool = True,
    context: Optional['DecompositionContext'] = None,
) -> Union[TDefault, List['cirq.Operation']]:
    pass


def decompose_once_with_qubits(
    val: Any,
    qubits: Iterable['cirq.Qid'],
    default=RaiseTypeErrorIfNotProvided,
    flatten: bool = True,
    context: Optional['DecompositionContext'] = None,
):
    """Decomposes a value into operations on the given qubits.

    This method is used when decomposing gates, which don't know which qubits
    they are being applied to unless told. It decomposes the gate exactly once,
    instead of decomposing it and then continuing to decomposing the decomposed
    operations recursively until some criteria is met.

    Args:
        val: The value to call `._decompose_(qubits)` on, if possible.
        qubits: The value to pass into the named `qubits` parameter of
            `val._decompose_`.
        default: A default result to use if the value doesn't have a
            `_decompose_` method or that method returns `NotImplemented` or
            `None`. If not specified, non-decomposable values cause a
            `TypeError`.
        flatten: If True, the returned OP-TREE will be flattened to a list of operations.
        context: Decomposition context specifying common configurable options for
            controlling the behavior of decompose.

    Returns:
        The result of `val._decompose_(qubits)`, if `val` has a
        `_decompose_` method and it didn't return `NotImplemented` or `None`.
        Otherwise `default` is returned, if it was specified. Otherwise an error
        is raised.

    TypeError:
        `val` didn't have a `_decompose_` method (or that method returned
        `NotImplemented` or `None`) and `default` wasn't set.
    """
    return decompose_once(val, default, tuple(qubits), flatten=flatten, context=context)


# pylint: enable=function-redefined


def _try_decompose_into_operations_and_qubits(
    val: Any,
) -> Tuple[Optional[List['cirq.Operation']], Sequence['cirq.Qid'], Tuple[int, ...]]:
    """Returns the value's decomposition (if any) and the qubits it applies to."""

    if isinstance(val, ops.Gate):
        # Gates don't specify qubits, and so must be handled specially.
        qid_shape = qid_shape_protocol.qid_shape(val)
        qubits: Sequence[cirq.Qid] = devices.LineQid.for_qid_shape(qid_shape)
        return decompose_once_with_qubits(val, qubits, None), qubits, qid_shape

    if isinstance(val, ops.Operation):
        qid_shape = qid_shape_protocol.qid_shape(val)
        return decompose_once(val, None), val.qubits, qid_shape

    result = decompose_once(val, None)
    if result is not None:
        qubit_set = set()
        qid_shape_dict: Dict[cirq.Qid, int] = defaultdict(lambda: 1)
        for op in result:
            for level, q in zip(qid_shape_protocol.qid_shape(op), op.qubits):
                qubit_set.add(q)
                qid_shape_dict[q] = max(qid_shape_dict[q], level)
        qubits = sorted(qubit_set)
        return result, qubits, tuple(qid_shape_dict[q] for q in qubits)
    return None, (), ()
