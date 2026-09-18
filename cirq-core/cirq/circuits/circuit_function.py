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

"""A named quantum circuit template with symbolic parameters."""

from __future__ import annotations

from collections.abc import Sequence, Set
from typing import Any, TYPE_CHECKING

import sympy

from cirq import _compat, protocols, value
from cirq.circuits.circuit import AbstractCircuit

if TYPE_CHECKING:
    import cirq
    from cirq.circuits.frozen_circuit import FrozenCircuit


@value.value_equality
class CircuitFunction:
    """A function that maps from a given set of parameters to a circuit.

    Note: This is an experimental class designed as part of a prototype
       for Cirq 2.0. The interface for this class is subject to change
       between versions.
    """

    def __init__(
        self,
        name: str,
        circuit: cirq.AbstractCircuit,
        function_params: Sequence[sympy.Symbol] | None = None,
    ) -> None:
        """Initializes a CircuitFunction.

        Args:
            name: The name of the function.
            circuit: The circuit body. If mutable, it will be frozen.
            function_params: Optional ordered sequence of parameters to the
                `CircuitFunction`. If None, defaults to all symbols in `circuit`.

        Raises:
            TypeError: If `name` is not a str, `circuit` is not an `AbstractCircuit`,
                or any item in `function_params` is not a `sympy.Symbol`.
            ValueError: If `name` is empty.
        """
        if not isinstance(name, str):
            raise TypeError(f"Function name must be a string, got: {type(name)!r}.")
        if not name:
            raise ValueError("Function name must be a non-empty string.")

        if not isinstance(circuit, AbstractCircuit):
            raise TypeError(f"Expected circuit of type AbstractCircuit, got: {type(circuit)!r}.")

        self._name = name
        self._circuit = circuit.freeze()

        if function_params is None:
            circuit_symbols = protocols.parameter_symbols(self._circuit)
            self._function_params = tuple(sorted(circuit_symbols, key=lambda s: s.name))
        else:
            self._function_params = tuple(function_params)
            if not all(isinstance(p, sympy.Symbol) for p in self._function_params):
                raise TypeError("All parameters must be sympy Symbols.")
            param_names = [p.name for p in self._function_params]
            if len(set(param_names)) != len(param_names):
                raise ValueError("CircuitFunctions may not have duplicate parameters.")

    @property
    def name(self) -> str:
        return self._name

    @property
    def circuit(self) -> FrozenCircuit:
        return self._circuit

    @property
    def function_params(self) -> tuple[sympy.Symbol, ...]:
        return self._function_params

    def all_qubits(self) -> frozenset[cirq.Qid]:
        return self._circuit.all_qubits()

    def _value_equality_values_(self) -> Any:
        return (self._name, self._circuit, self._function_params)

    def _is_parameterized_(self) -> bool:
        """Returns true if the circuit includes non-function parameters."""
        return bool(self._parameter_names_())

    def _parameter_names_(self) -> Set[str]:
        """Returns names of the non-function parameters."""
        circuit_parameters = protocols.parameter_names(self._circuit)
        return circuit_parameters - {p.name for p in self._function_params}

    def _json_dict_(self) -> dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['name', 'circuit', 'function_params'])

    @classmethod
    def _from_json_dict_(
        cls,
        name: str,
        circuit: cirq.FrozenCircuit,
        function_params: Sequence[sympy.Symbol],
        **kwargs,
    ) -> CircuitFunction:
        return cls(name=name, circuit=circuit, function_params=function_params)

    def __call__(self, *args: cirq.TParamVal, **kwargs: cirq.TParamVal) -> cirq.FrozenCircuit:
        """Call the circuit function with given parameters values."""

        param_dict = dict(zip(self._function_params, args)) | kwargs
        positional_param_names = {p.name for p in self._function_params[: len(args)]}
        expected_param_names = {p.name for p in self._function_params}

        if not positional_param_names.isdisjoint(kwargs.keys()):
            raise TypeError(f"CircuitFunction {self.name} called with duplicate parameters.")
        if len(args) + len(kwargs) != len(self._function_params):
            raise TypeError(
                f"CircuitFunction {self.name} takes {len(self.function_params)}"
                f" parameters but {len(args) + len(kwargs)} were provided."
            )
        if not set(kwargs.keys()) <= expected_param_names:
            raise TypeError(
                f"CircuitFunction {self.name} called with unrecognized keyword arguments."
            )

        return protocols.resolve_parameters(self._circuit, param_dict)

    def __repr__(self) -> str:
        param_items = ", ".join(_compat.proper_repr(p) for p in self._function_params)
        return (
            f"cirq.CircuitFunction("
            f"name={self._name!r}, "
            f"circuit={self._circuit!r}, "
            f"function_params=[{param_items}]"
            f")"
        )

    def __str__(self) -> str:
        params_str = ", ".join(str(p) for p in self._function_params)
        return f"{self._name}({params_str})"
