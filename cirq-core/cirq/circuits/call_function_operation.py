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
from cirq.circuits.circuit_operation import CircuitOperation

if TYPE_CHECKING:
    import cirq


@value.value_equality
class CallFunctionOperation(ops.Operation):
    """An operation that maps to a Circuit upon _decompose_ via CircuitFunction.

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
        qubit_map: Mapping[cirq.Qid, cirq.Qid] | None = None,
        extern_keys: frozenset[cirq.MeasurementKey] = frozenset(),
    ) -> None:
        """Initializes a CallFunctionOperation.

        Args:
            circuit_function: The CircuitFunction to call.
            function_args: Positional argument values to pass to the CircuitFunction.
            param_resolver: Optional resolver for free parameters in the CircuitFunction.
                The resolver is applied once, after `function_args` are bound, to any
                remaining symbols (including symbols introduced via `function_args`).
                Entries for symbols that cannot appear in the called circuit are dropped.
            measurement_key_map: Remappings for measurement keys in the called circuit.
            parent_path: Measurement keys produced by the called circuit are
                prefixed with this path.
            qubit_map: Remappings for qubits in the called circuit, keyed by the qubits as they
                appear in `circuit_function.circuit` (i.e. before any `VariableQid` is resolved).
                Usually populated via `with_qubits` rather than directly.
            extern_keys: The set of measurement keys defined at extern scope. The values here
                are used by decomposition and simulation routines to cache which external
                measurement keys exist as possible binding targets for unbound
                `ClassicallyControlledOperation` keys. This field is not intended to be set or
                changed manually, and should be empty in circuits that aren't in the middle of
                decomposition.

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
        params = circuit_function.function_params
        if len(function_args) != len(params):
            raise ValueError(
                f"Expected {len(params)} parameters "
                f"for {circuit_function.name}, but got {len(function_args)}."
            )

        self._circuit_function = circuit_function
        self._circuit_operation = CircuitOperation(
            circuit_function.circuit,
            param_resolver=dict(zip(params, function_args)),
            measurement_key_map=dict(measurement_key_map or {}),
            parent_path=parent_path,
            qubit_map=dict(qubit_map or {}),
            extern_keys=extern_keys,
        ).with_params(param_resolver)

    def _with_circuit_operation(
        self, circuit_operation: cirq.CircuitOperation
    ) -> CallFunctionOperation:
        """Returns a new CallFunctionOperation with a different CircuitOperation"""
        op = type(self).__new__(type(self))
        op._circuit_function = self._circuit_function
        op._circuit_operation = circuit_operation
        return op

    @property
    def circuit_function(self) -> cirq.CircuitFunction:
        return self._circuit_function

    @property
    def function_args(self) -> tuple[cirq.TParamVal, ...]:
        """The arguments bound to `circuit_function.function_params`, after any resolution.

        Any unused args of the CircuitFunction are dropped.
        """
        resolver = self._circuit_operation.param_resolver
        return tuple(
            resolver.value_of(p, recursive=False) for p in self._circuit_function.function_params
        )

    @property
    def param_resolver(self) -> cirq.ParamResolver:
        """The resolver for symbols that are free in the called circuit."""
        bound = {p.name for p in self._circuit_function.function_params}
        return study.ParamResolver(
            {
                k: v
                for k, v in self._circuit_operation.param_resolver.param_dict.items()
                if (k.name if isinstance(k, sympy.Symbol) else k) not in bound
            }
        )

    @property
    def measurement_key_map(self) -> Mapping[str, str]:
        return self._circuit_operation.measurement_key_map

    @property
    def parent_path(self) -> tuple[str, ...]:
        return self._circuit_operation.parent_path

    @property
    def qubit_map(self) -> Mapping[cirq.Qid, cirq.Qid]:
        return self._circuit_operation.qubit_map

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        """The qubits this operation acts on after any resolution.

        Note that qubit maps are not applied to VariableQids currently.
        """
        qubit_map = self._circuit_operation.qubit_map
        resolver = self._circuit_operation.param_resolver
        return tuple(
            sorted(
                {
                    (
                        protocols.resolve_parameters(q, resolver, recursive=False)
                        if isinstance(q, ops.VariableQid)
                        else qubit_map.get(q, q)
                    )
                    for q in self._circuit_function.all_qubits()
                }
            )
        )

    def with_qubits(self, *new_qubits: cirq.Qid) -> CallFunctionOperation:
        current = self.qubits
        if tuple(new_qubits) == current:
            return self
        if len(new_qubits) != len(current):
            raise ValueError(f'Expected {len(current)} qubits, got {len(new_qubits)}.')
        # Note: CircuitOperation qubit maps don't work with variableqids currently
        return self._with_circuit_operation(
            self._circuit_operation.with_qubit_mapping(dict(zip(current, new_qubits)))
        )

    def _parameter_names_(self) -> Set[str]:
        resolver = self._circuit_operation.param_resolver
        return {
            name
            for circuit_param_name in protocols.parameter_names(self._circuit_function.circuit)
            for name in protocols.parameter_names(
                resolver.value_of(circuit_param_name, recursive=False)
            )
        }

    def _is_parameterized_(self) -> bool:
        return bool(self._parameter_names_())

    def _resolve_parameters_(
        self, resolver: cirq.ParamResolver, recursive: bool
    ) -> CallFunctionOperation:
        return self._with_circuit_operation(
            self._circuit_operation.with_params(resolver.param_dict, recursive)
        )

    def _has_unitary_(self) -> bool:
        """Returns false if has parameters/measurement, else resorts to fallback strategies."""
        if self._is_parameterized_() or self._is_measurement_():
            return False
        return NotImplemented

    def _control_keys_(self) -> frozenset[cirq.MeasurementKey]:
        """Returns the set of control keys used by the generated circuit.

        This is implemented manually as the underlying CircuitOperation
        implementation would decompose the template circuit, creating a copy.
        """
        keys = protocols.control_keys(self._circuit_function)
        if not keys:
            return keys
        keys = frozenset(
            protocols.with_measurement_key_mapping(k, self._circuit_operation.measurement_key_map)
            for k in keys
        )
        return frozenset(
            protocols.with_rescoped_keys(
                value.KeyCondition(k),
                self._circuit_operation.parent_path,
                self._circuit_operation._extern_keys,
            ).key
            for k in keys
        )

    def _decompose_(self) -> cirq.OP_TREE:
        return self._circuit_operation.replace().mapped_circuit()

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self._circuit_operation)

    def _measurement_key_objs_(self) -> frozenset[cirq.MeasurementKey]:
        return protocols.measurement_key_objs(self._circuit_operation)

    def _measurement_key_names_(self) -> frozenset[str]:
        return protocols.measurement_key_names(self._circuit_operation)

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> CallFunctionOperation:
        mapped = protocols.with_measurement_key_mapping(self._circuit_operation.replace(), key_map)
        return self._with_circuit_operation(mapped.replace())

    def _with_key_path_(self, path: tuple[str, ...]) -> CallFunctionOperation:
        return self._with_circuit_operation(protocols.with_key_path(self._circuit_operation, path))

    def _with_key_path_prefix_(self, prefix: tuple[str, ...]) -> CallFunctionOperation:
        return self._with_circuit_operation(
            protocols.with_key_path_prefix(self._circuit_operation, prefix)
        )

    def _with_rescoped_keys_(
        self, path: tuple[str, ...], bindable_keys: frozenset[cirq.MeasurementKey]
    ) -> CallFunctionOperation:
        return self._with_circuit_operation(
            protocols.with_rescoped_keys(self._circuit_operation, path, bindable_keys)
        )

    def _value_equality_values_(self) -> Any:
        return (self._circuit_function, self._circuit_operation)

    def _json_dict_(self) -> dict[str, Any]:
        return {
            'circuit_function': self._circuit_function,
            'function_args': list(self.function_args),
            'param_resolver': self.param_resolver,
            'measurement_key_map': self._circuit_operation.measurement_key_map,
            'parent_path': self._circuit_operation.parent_path,
            'qubit_map': sorted(self._circuit_operation.qubit_map.items()),
        }

    @classmethod
    def _from_json_dict_(
        cls,
        circuit_function: cirq.CircuitFunction,
        function_args: Sequence[cirq.TParamVal],
        param_resolver: cirq.ParamResolver,
        measurement_key_map: Mapping[str, str] | None = None,
        parent_path: Sequence[str] = (),
        qubit_map: Sequence[tuple[cirq.Qid, cirq.Qid]] = (),
        **kwargs: Any,
    ) -> CallFunctionOperation:
        return cls(
            circuit_function,
            function_args,
            param_resolver=param_resolver,
            measurement_key_map=measurement_key_map,
            parent_path=tuple(parent_path),
            qubit_map=dict(qubit_map),
        )

    def __repr__(self) -> str:
        function_args = self.function_args
        args = ", ".join(proper_repr(a) for a in function_args)
        if len(function_args) == 1:
            args += ","
        result = (
            f"cirq.CallFunctionOperation("
            f"circuit_function={proper_repr(self._circuit_function)}, "
            f"function_args=({args}), "
            f"param_resolver={proper_repr(self.param_resolver.param_dict)}"
        )
        if self._circuit_operation.measurement_key_map:
            result += (
                f", measurement_key_map={proper_repr(self._circuit_operation.measurement_key_map)}"
            )
        if self._circuit_operation.parent_path:
            result += f", parent_path={proper_repr(self._circuit_operation.parent_path)}"
        if self._circuit_operation.qubit_map:
            result += f", qubit_map={proper_repr(self._circuit_operation.qubit_map)}"
        return result + ")"

    def __str__(self) -> str:
        args = ", ".join(str(p) for p in self.function_args)
        result = f"{self._circuit_function.name}({args})"
        env = self.param_resolver
        if env:
            resolver_items = ", ".join(
                f"{k}={env.value_of(k, recursive=False)}"
                for k in sorted(env.param_dict.keys(), key=lambda ki: str(ki))
            )
            result += f" [{resolver_items}]"
        if self._circuit_operation.measurement_key_map:
            key_items = ", ".join(
                f"{k}: {v}" for k, v in sorted(self._circuit_operation.measurement_key_map.items())
            )
            result += f" key_map={{{key_items}}}"
        if self._circuit_operation.parent_path:
            result += f" parent_path={self._circuit_operation.parent_path}"
        if self._circuit_operation.qubit_map:
            qubit_items = ", ".join(
                f"{k}: {v}" for k, v in sorted(self._circuit_operation.qubit_map.items())
            )
            result += f" qubit_map={{{qubit_items}}}"
        return result

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        formatted_args = ", ".join(str(p) for p in self.function_args)
        label = f"{self._circuit_function.name}({formatted_args})"
        wire_symbols: tuple[str, ...] = ()
        if self.qubits:
            wire_symbols = (label,) + ("",) * (len(self.qubits) - 1)
        if args.label_map is not None:
            wire_symbols += ("^",) * len(protocols.control_keys(self))
        return protocols.CircuitDiagramInfo(wire_symbols=wire_symbols)
