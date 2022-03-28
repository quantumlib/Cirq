# Copyright 2022 The Cirq Developers
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

"""Defines the API for circuit transformers in Cirq."""

import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
    cast,
    Any,
    Callable,
    Tuple,
    Hashable,
    List,
    overload,
    Optional,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from typing_extensions import Protocol

from cirq import circuits

if TYPE_CHECKING:
    import cirq


class LogLevel(enum.Enum):
    """Different logging resolution options for `cirq.TransformerLogger`.

    The enum values of the logging levels are used to filter the stored logs when printing.
    In general, a logging level `X` includes all logs stored at a level >= 'X'.

    Args:
        ALL:     All levels. Used to filter logs when printing.
        DEBUG:   Designates fine-grained informational events that are most useful to debug /
                 understand in-depth any unexpected behavior of the transformer.
        INFO:    Designates informational messages that highlight the actions of a transformer.
        WARNING: Designates unwanted or potentially harmful situations.
        NONE:    No levels. Used to filter logs when printing.
    """

    ALL = 0
    DEBUG = 10
    INFO = 20
    WARNING = 30
    NONE = 40


@dataclasses.dataclass
class _LoggerNode:
    """Stores logging data of a single transformer stage in `cirq.TransformerLogger`.

    The class is used to define a logging graph to store logs of sequential or nested transformers.
    Each node corresponds to logs of a single transformer stage.

    Args:
        transformer_id:   Integer specifying a unique id for corresponding transformer stage.
        transformer_name: Name of the corresponding transformer stage.
        initial_circuit:  Initial circuit before the transformer stage began.
        final_circuit:    Final circuit after the transformer stage ended.
        logs:             Messages logged by the transformer stage.
        nested_loggers:   `transformer_id`s of nested transformer stages which were called by
                          the current stage.
    """

    transformer_id: int
    transformer_name: str
    initial_circuit: 'cirq.AbstractCircuit'
    final_circuit: 'cirq.AbstractCircuit'
    logs: List[Tuple[LogLevel, Tuple[str, ...]]] = dataclasses.field(default_factory=list)
    nested_loggers: List[int] = dataclasses.field(default_factory=list)


class TransformerLogger:
    """Base Class for transformer logging infrastructure. Defaults to text-based logging.

    The logger implementation should be stateful, s.t.:
        - Each call to `register_initial` registers a new transformer stage and initial circuit.
        - Each subsequent call to `log` should store additional logs corresponding to the stage.
        - Each call to `register_final` should register the end of the currently active stage.

    The logger assumes that
        - Transformers are run sequentially.
        - Nested transformers are allowed, in which case the behavior would be similar to a
          doing a depth first search on the graph of transformers -- i.e. the top level transformer
          would end (i.e. receive a `register_final` call) once all nested transformers (i.e. all
          `register_initial` calls received while the top level transformer was active) have
          finished (i.e. corresponding `register_final` calls have also been received).
        - This behavior can be simulated by maintaining a stack of currently active stages and
          adding data from `log` calls to the stage at the top of the stack.

    The `LogLevel`s can be used to control the input processing and output resolution of the logs.
    """

    def __init__(self):
        """Initializes TransformerLogger."""
        self._curr_id: int = 0
        self._logs: List[_LoggerNode] = []
        self._stack: List[int] = []

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        """Register the beginning of a new transformer stage.

        Args:
            circuit: Input circuit to the new transformer stage.
            transformer_name: Name of the new transformer stage.
        """
        if self._stack:
            self._logs[self._stack[-1]].nested_loggers.append(self._curr_id)
        self._logs.append(_LoggerNode(self._curr_id, transformer_name, circuit, circuit))
        self._stack.append(self._curr_id)
        self._curr_id += 1

    def log(self, *args: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log additional metadata corresponding to the currently active transformer stage.

        Args:
            *args: The additional metadata to log.
            level: Logging level to control the amount of metadata that gets put into the context.

        Raises:
            ValueError: If there's no active transformer on the stack.
        """
        if len(self._stack) == 0:
            raise ValueError('No active transformer found.')
        self._logs[self._stack[-1]].logs.append((level, args))

    def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        """Register the end of the currently active transformer stage.

        Args:
            circuit: Final transformed output circuit from the transformer stage.
            transformer_name: Name of the (currently active) transformer stage which ends.

        Raises:
            ValueError: If `transformer_name` is different from currently active transformer name.
        """
        tid = self._stack.pop()
        if self._logs[tid].transformer_name != transformer_name:
            raise ValueError(
                f"Expected `register_final` call for currently active transformer "
                f"{self._logs[tid].transformer_name}."
            )
        self._logs[tid].final_circuit = circuit

    def show(self, level: LogLevel = LogLevel.INFO) -> None:
        """Show the stored logs >= level in the desired format.

        Args:
            level: The logging level to filter the logs with. The method shows all logs with a
            `LogLevel` >= `level`.
        """

        def print_log(log: _LoggerNode, pad=''):
            print(pad, f"Transformer-{1+log.transformer_id}: {log.transformer_name}", sep='')
            print(pad, "Initial Circuit:", sep='')
            print(textwrap.indent(str(log.initial_circuit), pad), "\n", sep='')
            for log_level, log_text in log.logs:
                if log_level.value >= level.value:
                    print(pad, log_level, *log_text)
            print("\n", pad, "Final Circuit:", sep='')
            print(textwrap.indent(str(log.final_circuit), pad))
            print("----------------------------------------")

        done = [0] * self._curr_id
        for i in range(self._curr_id):
            # Iterative DFS.
            stack = [(i, '')] if not done[i] else []
            while len(stack) > 0:
                log_id, pad = stack.pop()
                print_log(self._logs[log_id], pad)
                done[log_id] = True
                for child_id in self._logs[log_id].nested_loggers[::-1]:
                    stack.append((child_id, pad + ' ' * 4))


class NoOpTransformerLogger(TransformerLogger):
    """All calls to this logger are a no-op"""

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def log(self, *args: str, level: LogLevel = LogLevel.INFO) -> None:
        pass

    def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def show(self, level: LogLevel = LogLevel.INFO) -> None:
        pass


@dataclasses.dataclass(frozen=True)
class TransformerContext:
    """Stores common configurable options for transformers.

    Args:
        logger: `cirq.TransformerLogger` instance, which is a stateful logger used for logging
                the actions of individual transformer stages. The same logger instance should be
                shared across different transformer calls.
        tags_to_ignore: Tuple of tags which should be ignored while applying transformations on a
                circuit. Transformers should not transform any operation marked with a tag that
                belongs to this tuple. Note that any instance of a Hashable type (like `str`,
                `cirq.VirtualTag` etc.) is a valid tag.
        deep: If true, the transformer should be recursively applied to all sub-circuits wrapped
                inside circuit operations.
    """

    logger: TransformerLogger = NoOpTransformerLogger()
    tags_to_ignore: Tuple[Hashable, ...] = ()
    deep: bool = False


class TRANSFORMER(Protocol):
    """Protocol class defining the Transformer API for circuit transformers in Cirq.

    Any callable that satisfies the `cirq.TRANSFORMER` contract, i.e. takes a `cirq.AbstractCircuit`
    and `cirq.TransformerContext` and returns a transformed `cirq.AbstractCircuit`, is a valid
    transformer in Cirq.

    Note that transformers can also accept additional arguments as `**kwargs`, with default values
    specified for each keyword argument. A transformer could be a function, for example:

    >>> def convert_to_cz(
    >>>    circuit: cirq.AbstractCircuit,
    >>>    *,
    >>>    context: Optional[cirq.TransformerContext] = None,
    >>>    atol: float = 1e-8,
    >>> ) -> cirq.Circuit:
    >>>     ...

    Or it could be a class that implements `__call__` with the same API, for example:

    >>> class ConvertToSqrtISwaps:
    >>>    def __init__(self):
    >>>        ...
    >>>    def __call__(
    >>>        self,
    >>>        circuit: cirq.AbstractCircuit,
    >>>        *,
    >>>        context: Optional[cirq.TransformerContext] = None,
    >>>    ) -> cirq.AbstractCircuit:
    >>>        ...
    """

    def __call__(
        self, circuit: 'cirq.AbstractCircuit', *, context: Optional[TransformerContext] = None
    ) -> 'cirq.AbstractCircuit':
        ...


_TRANSFORMER_T = TypeVar('_TRANSFORMER_T', bound=TRANSFORMER)
_TRANSFORMER_CLS_T = TypeVar('_TRANSFORMER_CLS_T', bound=Type[TRANSFORMER])
_TRANSFORMER_OR_CLS_T = TypeVar(
    '_TRANSFORMER_OR_CLS_T', bound=Union[TRANSFORMER, Type[TRANSFORMER]]
)


@overload
def transformer(
    *, add_deep_support: bool = False
) -> Callable[[_TRANSFORMER_OR_CLS_T], _TRANSFORMER_OR_CLS_T]:
    pass


@overload
def transformer(cls_or_func: _TRANSFORMER_T, *, add_deep_support: bool = False) -> _TRANSFORMER_T:
    pass


@overload
def transformer(
    cls_or_func: _TRANSFORMER_CLS_T, *, add_deep_support: bool = False
) -> _TRANSFORMER_CLS_T:
    pass


def transformer(cls_or_func: Any = None, *, add_deep_support: bool = False) -> Any:
    """Decorator to verify API and append logging functionality to transformer functions & classes.

    A transformer is a callable that takes as inputs a `cirq.AbstractCircuit` and
    `cirq.TransformerContext`, and returns another `cirq.AbstractCircuit` without
    modifying the input circuit. A transformer could be a function, for example:

    >>> @cirq.transformer
    >>> def convert_to_cz(
    >>>    circuit: cirq.AbstractCircuit, *, context: Optional[cirq.TransformerContext] = None
    >>> ) -> cirq.Circuit:
    >>>    ...

    Or it could be a class that implements `__call__` with the same API, for example:

    >>> @cirq.transformer
    >>> class ConvertToSqrtISwaps:
    >>>    def __init__(self):
    >>>        ...
    >>>    def __call__(
    >>>        self,
    >>>        circuit: cirq.AbstractCircuit,
    >>>        *,
    >>>        context: Optional[cirq.TransformerContext] = None,
    >>>    ) -> cirq.Circuit:
    >>>        ...

    Note that transformers which take additional parameters as `**kwargs`, with default values
    specified for each keyword argument, are also supported. For example:

    >>> @cirq.transformer
    >>> def convert_to_sqrt_iswap(
    >>>     circuit: cirq.AbstractCircuit,
    >>>     *,
    >>>     context: Optional[cirq.TransformerContext] = None,
    >>>     atol: float = 1e-8,
    >>>     sqrt_iswap_gate: cirq.ISwapPowGate = cirq.SQRT_ISWAP_INV,
    >>>     cleanup_operations: bool = True,
    >>> ) -> cirq.Circuit:
    >>>     pass

    Args:
        cls_or_func: The callable class or function to be decorated.
        add_deep_support: If True, the decorator adds the logic to first apply the
            decorated transformer on subcircuits wrapped inside `cirq.CircuitOperation`s
            before applying it on the top-level circuit, if context.deep is True.

    Returns:
        Decorated class / function which includes additional logging boilerplate.
    """

    # If keyword arguments were specified, python invokes the decorator method
    # without a `cls` argument, then passes `cls` into the result.
    if cls_or_func is None:
        return lambda deferred_cls_or_func: transformer(
            deferred_cls_or_func,
            add_deep_support=add_deep_support,
        )

    if isinstance(cls_or_func, type):
        cls = cls_or_func
        method = cls.__call__
        default_context = _get_default_context(method)

        @functools.wraps(method)
        def method_with_logging(
            self, circuit: 'cirq.AbstractCircuit', **kwargs
        ) -> 'cirq.AbstractCircuit':
            return _transform_and_log(
                add_deep_support,
                lambda circuit, **kwargs: method(self, circuit, **kwargs),
                cls.__name__,
                circuit,
                kwargs.get('context', default_context),
                **kwargs,
            )

        setattr(cls, '__call__', method_with_logging)
        return cls
    else:
        assert callable(cls_or_func)
        func = cls_or_func
        default_context = _get_default_context(func)

        @functools.wraps(func)
        def func_with_logging(circuit: 'cirq.AbstractCircuit', **kwargs) -> 'cirq.AbstractCircuit':
            return _transform_and_log(
                add_deep_support,
                func,
                func.__name__,
                circuit,
                kwargs.get('context', default_context),
                **kwargs,
            )

        return func_with_logging


def _get_default_context(func: TRANSFORMER) -> TransformerContext:
    sig = inspect.signature(func)
    default_context = sig.parameters["context"].default
    assert (
        default_context != inspect.Parameter.empty
    ), "`context` argument must have a default value specified."
    return default_context


def _run_transformer_on_circuit(
    add_deep_support: bool,
    func: TRANSFORMER,
    circuit: 'cirq.AbstractCircuit',
    extracted_context: Optional[TransformerContext],
    **kwargs,
) -> 'cirq.AbstractCircuit':
    mutable_circuit = None
    if extracted_context and extracted_context.deep and add_deep_support:
        batch_replace = []
        for i, op in circuit.findall_operations(
            lambda o: isinstance(o.untagged, circuits.CircuitOperation)
        ):
            op_untagged = cast(circuits.CircuitOperation, op.untagged)
            if not set(op.tags).isdisjoint(extracted_context.tags_to_ignore):
                continue
            op_untagged = op_untagged.replace(
                circuit=_run_transformer_on_circuit(
                    add_deep_support, func, op_untagged.circuit, extracted_context, **kwargs
                ).freeze()
            )
            batch_replace.append((i, op, op_untagged.with_tags(*op.tags)))
        mutable_circuit = circuit.unfreeze(copy=True)
        mutable_circuit.batch_replace(batch_replace)
    return func(mutable_circuit if mutable_circuit else circuit, **kwargs)


def _transform_and_log(
    add_deep_support: bool,
    func: TRANSFORMER,
    transformer_name: str,
    circuit: 'cirq.AbstractCircuit',
    extracted_context: Optional[TransformerContext],
    **kwargs,
) -> 'cirq.AbstractCircuit':
    """Helper to log initial and final circuits before and after calling the transformer."""
    if extracted_context:
        extracted_context.logger.register_initial(circuit, transformer_name)
    transformed_circuit = _run_transformer_on_circuit(
        add_deep_support, func, circuit, extracted_context, **kwargs
    )
    if extracted_context:
        extracted_context.logger.register_final(transformed_circuit, transformer_name)
    return transformed_circuit
