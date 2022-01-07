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

import abc
import functools
from typing import (
    Callable,
    Tuple,
    TypeVar,
    Hashable,
    Optional,
    cast,
    Type,
    overload,
)
import dataclasses
import enum

from cirq.circuits.circuit import CIRCUIT_TYPE
from cirq.circuits import circuit, frozen_circuit


class LogLevel(enum.Enum):
    """Different logging resolution options for `TransformerStatsLogger`."""

    INFO = 1
    VERBOSE = 2
    WARNING = 3


class TransformerStatsLoggerBase:
    """Abstract Base Class for transformer logging infrastructure.

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

    @abc.abstractmethod
    def register_initial(self, circuit: CIRCUIT_TYPE, transformer_name: str) -> None:
        """Register the beginning of a new transformer stage.

        Args:
            circuit: Input circuit to the new transformer stage.
            transformer_name: Name of the new transformer stage.
        """

    @abc.abstractmethod
    def log(self, *args: str, level: LogLevel = LogLevel.INFO) -> None:
        """Log additional metadata corresponding to the currently active transformer stage.

        Args:
            *args: The additional metadata to log.
            level: Logging level to control the amount of metadata that gets put into the context.
        """

    @abc.abstractmethod
    def register_final(self, circuit: CIRCUIT_TYPE, transformer_name: str) -> None:
        """Register the end of the currently active transformer stage.

        Args:
            circuit: Final transformed output circuit from the transformer stage.
            transformer_name: Name of the (currently active) transformer stage which ends.
        """


@dataclasses.dataclass()
class TransformerContext:
    """Stores common configurable options for transformers."""

    logger: Optional[TransformerStatsLoggerBase] = None
    ignore_tags: Tuple[Hashable, ...] = ()
    max_parallelism: int = 1


TRANSFORMER_TYPE = Callable[[CIRCUIT_TYPE, TransformerContext], CIRCUIT_TYPE]
TRANSFORMER_TYPE_T = TypeVar(
    'TRANSFORMER_TYPE_T',
    TRANSFORMER_TYPE[circuit.Circuit],
    TRANSFORMER_TYPE[frozen_circuit.FrozenCircuit],
)
TRANSFORMER_CLS_TYPE = TypeVar(
    'TRANSFORMER_CLS_TYPE',
    Type[TRANSFORMER_TYPE[frozen_circuit.FrozenCircuit]],
    Type[TRANSFORMER_TYPE[circuit.Circuit]],
)


def _transformer_with_logging(
    func: TRANSFORMER_TYPE[CIRCUIT_TYPE],
    transformer_name: str,
    circuit: CIRCUIT_TYPE,
    context: TransformerContext,
) -> CIRCUIT_TYPE:
    """Helper to append logging functionality to transformers."""

    if context.logger is not None:
        context.logger.register_initial(circuit, transformer_name)
    circuit = func(circuit, context)
    if context.logger is not None:
        context.logger.register_final(circuit, transformer_name)
    return circuit


def transformer_method(func: TRANSFORMER_TYPE[CIRCUIT_TYPE]) -> TRANSFORMER_TYPE[CIRCUIT_TYPE]:
    """Decorator to verify API and append logging functionality to callable transformer methods.

    For example:
    >>> @transformer_method
    >>> def convert_to_cz(circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
    >>>    ...

    Args:
        func: Transformer implemented as a method from (CIRCUIT_TYPE, Context) --> CIRCUIT_TYPE.

    Returns:
        Method corresponding to original transformers + additional logging boilerplate.
    """

    @functools.wraps(func)
    def transformer_with_logging_func(
        circuit: CIRCUIT_TYPE, context: TransformerContext
    ) -> CIRCUIT_TYPE:
        return _transformer_with_logging(func, func.__name__, circuit, context)

    return cast(TRANSFORMER_TYPE[CIRCUIT_TYPE], transformer_with_logging_func)


def transformer_class(cls: TRANSFORMER_CLS_TYPE) -> TRANSFORMER_CLS_TYPE:
    """Decorator to verify API and append logging functionality to callable transformer class.

    The decorated class must implement the `__call__` method satisfying the `TRANSFORMER_TYPE` API.

    For example:
    >>> @transformer_class
    >>> class ConvertToSqrtISwaps:
    >>>    def __init__(self):
    >>>        ...
    >>>    def __call__(circuit: cirq.Circuit, context: cirq.TransformerContext) -> cirq.Circuit:
    >>>        ...

    Args:
        cls: Transformer implemented as a class overriding the `__call__` method, which satisfies
            the `TRANSFORMER_TYPE` API.

    Returns:
        `cls` with `cls.__call__` updated to include additional logging boilerplate.
    """

    old_func = cls.__call__

    def transformer_with_logging_cls(
        self: TRANSFORMER_TYPE, circuit: CIRCUIT_TYPE, context: TransformerContext
    ) -> CIRCUIT_TYPE:
        def call_old_func(c: CIRCUIT_TYPE, ct: TransformerContext) -> CIRCUIT_TYPE:
            return old_func(self, c, ct)

        return _transformer_with_logging(call_old_func, cls.__name__, circuit, context)

    setattr(cls, '__call__', transformer_with_logging_cls)
    return cls


TRANSFORMER_UNION_TYPE = TypeVar(
    'TRANSFORMER_UNION_TYPE',
    TRANSFORMER_TYPE[frozen_circuit.FrozenCircuit],
    TRANSFORMER_TYPE[circuit.Circuit],
    Type[TRANSFORMER_TYPE[frozen_circuit.FrozenCircuit]],
    Type[TRANSFORMER_TYPE[circuit.Circuit]],
)


@overload
def transformer(cls_or_func: TRANSFORMER_TYPE_T) -> TRANSFORMER_TYPE_T:
    pass


@overload
def transformer(cls_or_func: TRANSFORMER_CLS_TYPE) -> TRANSFORMER_CLS_TYPE:
    pass


def transformer(
    cls_or_func: TRANSFORMER_UNION_TYPE,
) -> TRANSFORMER_UNION_TYPE:
    """Decorator to verify API and append logging functionality to transformer methods & classes.

    The decorated method must satisfy the `TRANSFORMER_TYPE` API or
    The decorated class must implement the `__call__` method which satisfies the
    `TRANSFORMER_TYPE` API.

    Args:
        cls_or_func: The callable class or method to be decorated.

    Returns:
        Decorated class / method which includes additional logging boilerplate.
    """
    if isinstance(cls_or_func, type):
        return transformer_class(cls_or_func)
    assert callable(
        cls_or_func
    ), f"The decorated object {cls_or_func} must be a callable class or a method."
    return transformer_method(cls_or_func)
