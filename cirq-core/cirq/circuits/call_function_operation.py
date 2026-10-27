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

from collections.abc import Mapping, Sequence, Set
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
        measurement_key_map: Mapping[str, str] | None = None,
        parent_path: tuple[str, ...] = (),
    ) -> None:
        """Initializes a CallFunctionOperation.

        Args:
            circuit_function: The CircuitFunction to call.
            function_args: Positional argument values to pass to the CircuitFunction.
            param_resolver: Optional resolver for free parameters in the CircuitFunction.
                The resolver is applied AFTER the CircuitFunction parameters are substituted.
            measurement_key_map: Remappings for measurement keys in the called circuit.
            parent_path: Measurement keys produced by the called circuit are
                prefixed with this path.

        Raises:
            TypeError: If `circuit_function` is not a `CircuitFunction`.
            ValueError: If the number of `function_args` does not match the number of
                `function_params` in `circuit_function`, or if `measurement_key_map` maps
                to an invalid key.
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

        # Only retain resolver entries that can affect the called circuit
        resolver = study.ParamResolver(param_resolver)
        possible_resolver_names = self._unresolved_parameter_names()
        filtered_resolver_items = {
            k: v
            for k, v in resolver.param_dict.items()
            if (k.name if isinstance(k, sympy.Symbol) else k) in possible_resolver_names
        }
        self._param_resolver = (
            resolver
            if len(filtered_resolver_items) == len(resolver.param_dict)
            else study.ParamResolver(filtered_resolver_items)
        )

        # resolve qubits if they are parameterized
        function_param_mapping = study.ParamResolver(
            dict(zip(self._circuit_function.function_params, self._function_args))
        )

        def resolve_qubit(q: cirq.Qid) -> cirq.Qid:
            if not isinstance(q, ops.VariableQid):
                return q
            q = protocols.resolve_parameters(q, function_param_mapping, recursive=False)
            return protocols.resolve_parameters(q, self._param_resolver, recursive=False)

        self._qubits = tuple(
            sorted({resolve_qubit(q) for q in self._circuit_function.all_qubits()})
        )

        self._measurement_key_map = dict(measurement_key_map or {})
        for mapped_key in self._measurement_key_map.values():
            if value.MEASUREMENT_KEY_SEPARATOR in mapped_key:
                raise ValueError(
                    f'Mapping to invalid key: {mapped_key}. "{value.MEASUREMENT_KEY_SEPARATOR}" '
                    'is not allowed for measurement keys in a CallFunctionOperation'
                )
        self._parent_path = tuple(parent_path)

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
    def measurement_key_map(self) -> Mapping[str, str]:
        return self._measurement_key_map

    @property
    def parent_path(self) -> tuple[str, ...]:
        return self._parent_path

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return self._qubits

    def replace(self, **changes: Any) -> CallFunctionOperation:
        """Returns a copy of this operation with the specified changes."""
        kwargs: dict[str, Any] = {
            'circuit_function': self._circuit_function,
            'function_args': self._function_args,
            'param_resolver': self._param_resolver,
            'measurement_key_map': self._measurement_key_map,
            'parent_path': self._parent_path,
            **changes,
        }
        return CallFunctionOperation(**kwargs)

    def with_qubits(self, *new_qubits: cirq.Qid) -> CallFunctionOperation:
        if tuple(new_qubits) == self._qubits:
            return self
        raise ValueError("CallFunctionOperation does not support `with_qubits`.")

    def _decompose_(self) -> cirq.OP_TREE:
        """Evaluates the CircuitFunction with function_args and applies all maps.

        The order is: function args, then `param_resolver`, then `measurement_key_map`, then
        key rescoping under `parent_path`.
        """
        circuit: cirq.AbstractCircuit = self._circuit_function(*self._function_args)
        if self._param_resolver:
            circuit = protocols.resolve_parameters(circuit, self._param_resolver, recursive=False)
        if self._measurement_key_map:
            circuit = protocols.with_measurement_key_mapping(circuit, self._measurement_key_map)
        if self._parent_path:
            circuit = protocols.with_rescoped_keys(circuit, self._parent_path)
        return circuit

    def _has_unitary_(self) -> bool:
        """Returns false if has parameters/measurement, else resorts to fallback strategies."""
        if self._is_parameterized_() or self._is_measurement_():
            return False
        return NotImplemented

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self._circuit_function)

    def _mapped_keys(self, keys):
        """Returns a set of keys after applying the measurement key mapping to them"""
        if self._measurement_key_map:
            keys = frozenset(
                protocols.with_measurement_key_mapping(k, self._measurement_key_map) for k in keys
            )
        return keys

    def _measurement_key_objs_(self) -> frozenset[cirq.MeasurementKey]:
        keys = self._mapped_keys(protocols.measurement_key_objs(self._circuit_function))
        if self._parent_path:
            keys = frozenset(k.with_key_path_prefix(*self._parent_path) for k in keys)
        return keys

    def _measurement_key_names_(self) -> frozenset[str]:
        return frozenset(str(k) for k in self._measurement_key_objs_())

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        return self._mapped_keys(protocols.control_keys(self._circuit_function))

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> CallFunctionOperation:
        """Returns a copy of this operation with `key_map` composed onto its key mapping.

        Raises:
            ValueError: If the composed mapping would merge two distinct measurement keys.
        """
        new_map = {}
        for k_obj in protocols.measurement_keys_touched(self._circuit_function):
            k = k_obj.name
            k_new = self._measurement_key_map.get(k, k)
            k_new = key_map.get(k_new, k_new)
            if k_new != k:
                new_map[k] = k_new
        new_op = self.replace(measurement_key_map=new_map)
        if len(protocols.measurement_keys_touched(new_op)) != len(
            protocols.measurement_keys_touched(self)
        ):
            raise ValueError(
                f'Collision in measurement key map composition. Original map:\n'
                f'{self._measurement_key_map}\nApplied changes: {key_map}'
            )
        return new_op

    def _with_key_path_(self, path: tuple[str, ...]) -> CallFunctionOperation:
        return self.replace(parent_path=path)

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> CallFunctionOperation:
        return self.replace(parent_path=prefix + self._parent_path)

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> CallFunctionOperation:
        return self.replace(parent_path=path + self._parent_path)

    def _value_equality_values_(self) -> Any:
        return (
            self._circuit_function,
            self._function_args,
            self._param_resolver,
            frozenset(self._measurement_key_map.items()),
            self._parent_path,
        )

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
        return self.replace(param_resolver=new_params)

    def _json_dict_(self) -> dict[str, Any]:
        return {
            'circuit_function': self._circuit_function,
            'function_args': list(self._function_args),
            'param_resolver': self._param_resolver,
            'measurement_key_map': self._measurement_key_map,
            'parent_path': self._parent_path,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        circuit_function: cirq.CircuitFunction,
        function_args: Sequence[cirq.TParamVal],
        param_resolver: cirq.ParamResolver,
        measurement_key_map: Mapping[str, str] | None = None,
        parent_path: Sequence[str] = (),
        **kwargs: Any,
    ) -> CallFunctionOperation:
        return cls(
            circuit_function,
            function_args,
            param_resolver=param_resolver,
            measurement_key_map=measurement_key_map,
            parent_path=tuple(parent_path),
        )

    def __repr__(self) -> str:
        args = ", ".join(proper_repr(a) for a in self._function_args)
        result = (
            f"cirq.CallFunctionOperation("
            f"circuit_function={proper_repr(self._circuit_function)}, "
            f"function_args=[{args}], "
            f"param_resolver={proper_repr(self._param_resolver.param_dict)}"
        )
        if self._measurement_key_map:
            result += f", measurement_key_map={proper_repr(self._measurement_key_map)}"
        if self._parent_path:
            result += f", parent_path={proper_repr(self._parent_path)}"
        return result + ")"

    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self._function_args)
        result = f"{self._circuit_function.name}({args})"
        if self._param_resolver:
            resolver_items = ", ".join(
                f"{k}={self._param_resolver.param_dict[k]}"
                for k in sorted(self._param_resolver.param_dict.keys(), key=lambda ki: str(ki))
            )
            result += f" [{resolver_items}]"
        if self._measurement_key_map:
            key_items = ", ".join(f"{k}: {v}" for k, v in self._measurement_key_map.items())
            result += f" key_map={{{key_items}}}"
        if self._parent_path:
            result += f" parent_path={self._parent_path}"
        return result

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        formatted_args = ", ".join(str(p) for p in self._function_args)
        label = f"{self._circuit_function.name}({formatted_args})"
        if not self.qubits:
            return protocols.CircuitDiagramInfo(wire_symbols=())
        return protocols.CircuitDiagramInfo(wire_symbols=(label,) + ("",) * (len(self.qubits) - 1))
