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

import textwrap
import functools
from typing import (
    Callable,
    Tuple,
    Hashable,
    List,
    cast,
    Type,
    overload,
    Union,
    TYPE_CHECKING,
)

import dataclasses
import enum

if TYPE_CHECKING:
    import cirq


class LogLevel(enum.Enum):
    """Different logging resolution options for `cirq.TransformerStatsLogger`.

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
class LoggerNode:
    """Stores logging data of a single transformer stage in `cirq.TransformerStatsLogger`."""

    transformer_id: int
    transformer_name: str
    initial_circuit: 'cirq.AbstractCircuit'
    final_circuit: 'cirq.AbstractCircuit'
    logs: List[Tuple[LogLevel, Tuple[str, ...]]] = dataclasses.field(default_factory=list)
    nested_loggers: List[int] = dataclasses.field(default_factory=list)


class TransformerStatsLogger:
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
        """Initializes TransformerStatsLogger."""
        self._curr_id: int = 0
        self._logs: List[LoggerNode] = []
        self._stack: List[int] = []

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        """Register the beginning of a new transformer stage.

        Args:
            circuit: Input circuit to the new transformer stage.
            transformer_name: Name of the new transformer stage.
        """
        if self._stack:
            self._logs[self._stack[-1]].nested_loggers.append(self._curr_id)
        self._logs.append(LoggerNode(self._curr_id, transformer_name, circuit, circuit))
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
        """Show the stored logs >= level in the desired format."""

        def print_log(log: LoggerNode, pad=''):
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


class NoOpTransformerStatsLogger(TransformerStatsLogger):
    """All calls to this logger are a no-op"""

    def register_initial(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def log(self, *args: str, level: LogLevel = LogLevel.INFO) -> None:
        pass

    def register_final(self, circuit: 'cirq.AbstractCircuit', transformer_name: str) -> None:
        pass

    def show(self, level: LogLevel = LogLevel.INFO) -> None:
        pass


@dataclasses.dataclass()
class TransformerContext:
    """Stores common configurable options for transformers.

    Args:
        logger: `cirq.TransformerStatsLogger` instance, which is a stateful logger used for logging
                the actions of individual transformer stages. The same logger instance should be
                shared across different transformer calls.
        ignore_tags: Tuple of tags which should be ignored while applying transformations on a
                circuit. Transformers should not transform any operation marked with a tag that
                belongs to this tuple. Note that any instance of a Hashable type (like `str`,
                `cirq.VirtualTag` etc.) is a valid tag.
    """

    logger: TransformerStatsLogger = NoOpTransformerStatsLogger()
    ignore_tags: Tuple[Hashable, ...] = ()


TRANSFORMER = Callable[['cirq.AbstractCircuit', TransformerContext], 'cirq.Circuit']
TRANSFORMER_TO_DECORATE = Callable[['cirq.Circuit', TransformerContext], 'cirq.Circuit']


def _transform_and_log(
    func: TRANSFORMER_TO_DECORATE,
    transformer_name: str,
    circuit: 'cirq.AbstractCircuit',
    context: TransformerContext,
) -> 'cirq.Circuit':
    """Helper to log initial and final circuits before and after calling the transformer."""

    context.logger.register_initial(circuit, transformer_name)
    transformed_circuit = func(circuit.unfreeze(copy=True), context)
    context.logger.register_final(transformed_circuit, transformer_name)
    return transformed_circuit


@overload
def transformer(cls_or_func: TRANSFORMER_TO_DECORATE) -> TRANSFORMER:
    pass


@overload
def transformer(cls_or_func: Type[TRANSFORMER_TO_DECORATE]) -> Type[TRANSFORMER]:
    pass


def transformer(
    cls_or_func: Union[TRANSFORMER_TO_DECORATE, Type[TRANSFORMER_TO_DECORATE]],
) -> Union[TRANSFORMER, Type[TRANSFORMER]]:
    """Decorator to verify API and append logging functionality to transformer functions & classes.

    The decorated function or class must satisfy
    `Callable[[cirq.Circuit, cirq.TransformerContext], cirq.Circuit]` API. For Example:

    >>> @cirq.transformer
    >>> def convert_to_cz(circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
    >>>    ...

    The decorated class must implement the `__call__` method to satisfy the above API.

    >>> @cirq.transformer
    >>> class ConvertToSqrtISwaps:
    >>>    def __init__(self):
    >>>        ...
    >>>    def __call__(
    >>>        self, circuit: cirq.Circuit, context: cirq.TransformerContext
    >>>    ) -> cirq.Circuit:
    >>>        ...

    Args:
        cls_or_func: The callable class or method to be decorated.

    Returns:
        Decorated class / method which includes additional logging boilerplate. The decorated
        callable always receives a copy of the input circuit so that the input is never mutated.
    """
    assert callable(
        cls_or_func
    ), f"The decorated object {cls_or_func} must be a callable class or a method."
    if isinstance(cls_or_func, type):
        old_func = cls_or_func.__call__

        def transformer_with_logging_cls(
            self: Type[TRANSFORMER],
            circuit: 'cirq.AbstractCircuit',
            context: TransformerContext,
        ) -> 'cirq.Circuit':
            def call_old_func(c: 'cirq.Circuit', ct: TransformerContext) -> 'cirq.Circuit':
                return old_func(self, c, ct)

            return _transform_and_log(call_old_func, cls_or_func.__name__, circuit, context)

        setattr(cls_or_func, '__call__', transformer_with_logging_cls)
        return cast(Type[TRANSFORMER], cls_or_func)
    else:
        func = cast(TRANSFORMER_TO_DECORATE, cls_or_func)

        @functools.wraps(func)
        def transformer_with_logging_func(
            circuit: 'cirq.AbstractCircuit',
            context: TransformerContext,
        ) -> 'cirq.Circuit':
            return _transform_and_log(func, func.__name__, circuit, context)

        return cast(TRANSFORMER, transformer_with_logging_func)
