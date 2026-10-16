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
from typing import Any, TYPE_CHECKING

import sympy

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
        function_args: Sequence[cirq.TParamVal] = (),
        param_resolver: study.ParamResolverOrSimilarType | None = None,
    ) -> None:
        """Initializes a CallFunctionOperation.

        Args:
            circuit_function: The CircuitFunction to call.
            function_args: Positional argument values to pass to the CircuitFunction.
            param_resolver: Optional resolver for free parameters in the CircuitFunction.

        Raises:
            TypeError: If `circuit_function` is not a `CircuitFunction`.
            ValueError: If the number of `function_args` does not match the number of
                `function_params` in `circuit_function`.
        """
        if not isinstance(circuit_function, CircuitFunction):
            raise TypeError(
                f"Expected circuit_function of type CircuitFunction, got: {type(circuit_function)}."
            )
        if len(function_args) != len(circuit_function.function_params):
            raise ValueError(
                f"Expected {len(circuit_function.function_params)} parameters "
                f"for {circuit_function.name}, but got {len(function_args)}."
            )

        self._circuit_function = circuit_function
        self._function_args = tuple(function_args)
        self._param_resolver = study.ParamResolver(param_resolver)

        # resolve qubits if they are parameterized
        function_param_mapping = dict(
            zip(self._circuit_function.function_params, self._function_args)
        )
        self._qubits = tuple(
            sorted(
                protocols.resolve_parameters(
                    protocols.resolve_parameters(q, function_param_mapping, recursive=False),
                    self._param_resolver,
                    recursive=False,
                )
                for q in self._circuit_function.all_qubits()
            )
        )

    @property
    def circuit_function(self) -> cirq.CircuitFunction:
        return self._circuit_function

    @property
    def function_args(self) -> tuple[cirq.TParamVal, ...]:
        return self._function_args

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
        """Evaluates the CircuitFunction with function_args and resolves params."""
        circuit = self._circuit_function(*self._function_args)
        if self._param_resolver:
            circuit = protocols.resolve_parameters(circuit, self._param_resolver, recursive=False)
        return circuit

    def _has_unitary_(self) -> bool:
        """Returns false if has parameters/measurement, else resorts to fallback strategies."""
        if self._is_parameterized_() or self._is_measurement_():
            return False
        return NotImplemented

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self._circuit_function)

    def _measurement_key_objs_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.measurement_key_objs(self._circuit_function)

    def _measurement_key_names_(self) -> frozenset[str]:
        return protocols.measurement_key_names(self._circuit_function)

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.control_keys(self._circuit_function)

    def _value_equality_values_(
        self,
    ) -> tuple[cirq.CircuitFunction, tuple[cirq.TParamVal, ...], cirq.ParamResolver]:
        return (self._circuit_function, self._function_args, self._param_resolver)

    def _is_parameterized_(self) -> bool:
        """Returns True if the called circuit still contains unresolved parameters."""
        return bool(self._parameter_names_())

    def _unresolved_parameter_names(self) -> set[str]:
        """Returns the parameters of the decomposed circuit before `param_resolver` is applied."""
        bound = {p.name for p in self._circuit_function.function_params}
        names = set(protocols.parameter_names(self._circuit_function.circuit)) - bound
        for arg in self._function_args:
            names |= protocols.parameter_names(arg)
        return names

    def _parameter_names_(self) -> Set[str]:
        """Returns the parameters of the decomposed circuit."""
        names: set[str] = set()
        for name in self._unresolved_parameter_names():
            names |= protocols.parameter_names(self._param_resolver.value_of(name, recursive=False))
        return names

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> CallFunctionOperation:
        """Combines the new `resolver` with the stored resolver."""
        new_params: dict[sympy.Symbol, cirq.TParamVal] = {}
        for name in self._unresolved_parameter_names():
            k = sympy.Symbol(name)
            v = self._param_resolver.value_of(k, recursive=False)
            v = protocols.resolve_parameters(v, resolver, recursive)
            if v != k:
                new_params[k] = v
        return CallFunctionOperation(
            self._circuit_function, self._function_args, param_resolver=new_params
        )

    def _json_dict_(self) -> dict[str, Any]:
        return {
            'circuit_function': self._circuit_function,
            'function_args': list(self._function_args),
            'param_resolver': self._param_resolver,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        circuit_function: cirq.CircuitFunction,
        function_args: Sequence[cirq.TParamVal],
        param_resolver: cirq.ParamResolver,
        **kwargs: Any,
    ) -> CallFunctionOperation:
        return cls(circuit_function, function_args, param_resolver=param_resolver)

    def __repr__(self) -> str:
        args = ", ".join(proper_repr(a) for a in self._function_args)
        return (
            f"cirq.CallFunctionOperation("
            f"circuit_function={proper_repr(self._circuit_function)}, "
            f"function_args=[{args}], "
            f"param_resolver={proper_repr(self._param_resolver.param_dict)})"
        )

    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self._function_args)
        base = f"{self._circuit_function.name}({args})"
        if not self._param_resolver:
            return base
        resolver_items = ", ".join(
            f"{k}={self._param_resolver[k]}"
            for k in sorted(self._param_resolver.param_dict.keys(), key=lambda ki: str(ki))
        )
        return f"{base} [{resolver_items}]"

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        formatted_args = ", ".join(str(p) for p in self._function_args)
        label = f"{self._circuit_function.name}({formatted_args})"
        if not self.qubits:
            return protocols.CircuitDiagramInfo(wire_symbols=())
        return protocols.CircuitDiagramInfo(wire_symbols=(label,) + ("",) * (len(self.qubits) - 1))
