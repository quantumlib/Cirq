# Copyright 2026 The Cirq Developers
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

"""An operation representing the invocation of a CircuitFunction template."""

from __future__ import annotations

from collections.abc import Sequence, Set
from typing import TYPE_CHECKING

from cirq import ops, protocols, study, value
from cirq._compat import proper_repr
from cirq.circuits.circuit_function import CircuitFunction

if TYPE_CHECKING:
    import cirq


@value.value_equality
class CallFunctionOperation(ops.Operation):
    """An operation that maps to a Circuit at runtime via CircuitFunction.

    Note: This is an experimental class designed as part of a prototype
    for Cirq 2.0. The interface for this class is subject to change
    between versions.
    """

    def __init__(
        self,
        circuit_function: cirq.CircuitFunction,
        function_params: Sequence[cirq.TParamVal] = (),
        param_resolver: study.ParamResolverOrSimilarType | None = None,
    ) -> None:
        """Initializes a CallFunctionOperation.

        Args:
            circuit_function: The CircuitFunction to call.
            function_params: Positional argument values to pass to the CircuitFunction.
            param_resolver: Optional resolver for free parameters in the CircuitFunction.

        Raises:
            TypeError: If `circuit_function` is not a `CircuitFunction`.
            ValueError: If the number of `function_params` does not match the number of
                `function_params` in `circuit_function`.
        """
        if not isinstance(circuit_function, CircuitFunction):
            raise TypeError(
                f"Expected circuit_function of type CircuitFunction, got: {type(circuit_function)}."
            )
        if len(function_params) != len(circuit_function.function_params):
            raise ValueError(
                f"Expected {len(circuit_function.function_params)} parameters "
                f"for {circuit_function.name}, but got {len(function_params)}."
            )

        self._circuit_function = circuit_function
        self._function_params = tuple(function_params)
        self._param_resolver = study.ParamResolver(param_resolver)
        self._qubits = tuple(sorted(self._circuit_function.all_qubits()))

    @property
    def circuit_function(self) -> cirq.CircuitFunction:
        return self._circuit_function

    @property
    def function_params(self) -> tuple[cirq.TParamVal, ...]:
        return self._function_params

    @property
    def param_resolver(self) -> cirq.ParamResolver:
        return self._param_resolver

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._qubits

    def with_qubits(self, *new_qubits: cirq.Qid) -> CallFunctionOperation:
        if tuple(new_qubits) == self._qubits:
            return self
        raise ValueError("CallFunctionOperation does not support `with_qubits`.")

    def _decompose_(self) -> cirq.OP_TREE:
        """Evaluates the CircuitFunction with function_params and resolves params."""
        circuit = self._circuit_function(*self._function_params)
        if self._param_resolver:
            circuit = protocols.resolve_parameters(circuit, self._param_resolver)
        return circuit

    def _value_equality_values_(
        self,
    ) -> tuple[cirq.CircuitFunction, tuple[cirq.TParamVal, ...], cirq.ParamResolver]:
        return (self._circuit_function, self._function_params, self._param_resolver)

    def _is_parameterized_(self) -> bool:
        """Returns True if there are parameters that are not function parameters."""
        return bool(self._parameter_names_())

    def _parameter_names_(self) -> Set[str]:
        """Returns all parameters that are not function parameters."""
        function_param_names = {p.name for p in self._circuit_function.function_params}
        return protocols.parameter_names(self._circuit_function.circuit) - function_param_names

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool = False
    ) -> CallFunctionOperation:
        """Stores parameter resolver to resolve mapped circuit in the future."""
        combined_resolver_dict = dict(self._param_resolver.param_dict)
        combined_resolver_dict.update(resolver.param_dict)
        return CallFunctionOperation(
            self._circuit_function, self._function_params, param_resolver=combined_resolver_dict
        )

    def __repr__(self) -> str:
        return (
            f"cirq.CallFunctionOperation("
            f"circuit_function={proper_repr(self._circuit_function)}, "
            f"function_params={proper_repr(self.function_params)}, "
            f"param_resolver={proper_repr(self._param_resolver.param_dict)})"
        )

    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self._function_params)
        base = f"{self._circuit_function.name}({args})"
        if not self._param_resolver:
            return base
        resolver_items = ", ".join(
            f"{k}={v}" for k, v in sorted(self._param_resolver.param_dict.items())
        )
        return f"{base} [{resolver_items}]"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        formatted_args = ", ".join(str(p) for p in self._function_params)
        label = f"{self._circuit_function.name}({formatted_args})"
        if not self.qubits:
            return protocols.CircuitDiagramInfo(wire_symbols=())
        return protocols.CircuitDiagramInfo(wire_symbols=(label,) + ("",) * (len(self.qubits) - 1))
