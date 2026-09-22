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

from collections import Counter
from collections.abc import Sequence
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
        circuit: AbstractCircuit,
        function_params: Sequence[sympy.Symbol] | None = None,
    ) -> None:
        """Initializes a CircuitFunction.

        Args:
            name: The name of the function.
            circuit: The circuit body. If mutable, it will be frozen.
            function_params: The sequence of sympy Symbols that specify the parameters
                of the CircuitFunction. If None, defaults to all symbols in `circuit`,
                sorted alphabetically by name.

        Raises:
            TypeError: If `name` is not a str, `circuit` is not an `AbstractCircuit`,
                or any item in `function_params` is not a `sympy.Symbol`.
            ValueError: If `name` is empty or if duplicate parameters are provided.
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
            non_symbols = [p for p in self._function_params if not isinstance(p, sympy.Symbol)]
            if non_symbols:
                details = ", ".join(f"{p} ({type(p).__name__})" for p in non_symbols)
                raise TypeError(f"All parameters must be sympy Symbols, got: {details}.")
            param_names = [p.name for p in self._function_params]
            duplicates = [p for p, count in Counter(param_names).items() if count > 1]
            if duplicates:
                raise ValueError(
                    f"Cannot create CircuitFunction {name} with duplicate parameters: "
                    f"{', '.join(duplicates)}."
                )

    @property
    def name(self) -> str:
        """Returns the name of the circuit function."""
        return self._name

    @property
    def circuit(self) -> FrozenCircuit:
        """Returns the unsubstituted circuit returned by the function."""
        return self._circuit

    @property
    def function_params(self) -> tuple[sympy.Symbol, ...]:
        """Returns a tuple of the symbols representing the function parameters."""
        return self._function_params

    def all_qubits(self) -> frozenset[cirq.Qid]:
        """Returns the set of all qubits in the circuit."""
        return self._circuit.all_qubits()

    def _value_equality_values_(self) -> Any:
        return (self._name, self._circuit, self._function_params)

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
        """Call the circuit function with the given parameter values.

        Args:
            args: The values of some/all function parameters to use.
            kwargs: The values of some/all function parameters provided as keywords.

        Returns:
            The circuit body with the function parameters substituted by the arg values.

        Raises:
            TypeError: If the CircuitFunction is called with duplicate values for a single
                parameter, the incorrect number of parameters is provided, or an unrecognized
                keyword argument is provided.
        """

        positional_param_names = {p.name for p in self._function_params[: len(args)]}
        expected_param_names = {p.name for p in self._function_params}

        duplicates = positional_param_names & kwargs.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise TypeError(
                f"CircuitFunction {self.name} called with duplicate parameters: {names}."
            )
        unrecognized = kwargs.keys() - expected_param_names
        if unrecognized:
            names = ", ".join(sorted(unrecognized))
            raise TypeError(
                f"CircuitFunction {self.name} called with unrecognized keyword arguments: {names}."
            )
        if len(args) + len(kwargs) != len(self._function_params):
            raise TypeError(
                f"CircuitFunction {self.name} takes {len(self._function_params)}"
                f" parameter(s) but received {len(args) + len(kwargs)}."
            )

        param_dict = dict(zip((p.name for p in self._function_params), args)) | kwargs
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
