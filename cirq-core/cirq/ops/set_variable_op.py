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

"""Operation to set variable values during circuit execution.

Currently, this operation is expected to only work during simulation.
"""

from __future__ import annotations

from collections.abc import Set
from typing import Any, TYPE_CHECKING

import sympy

from cirq import protocols, value
from cirq.ops import raw_types
from cirq.study import resolver

if TYPE_CHECKING:
    import cirq


@value.value_equality
class SetVariable(raw_types.Operation):
    """An operation that assigns the value of a SymPy expression to a SymPy variable.

    This operation modifies the parameter resolver in the simulator's state.

    Note: This is an experimental feature for early fault-tolerant support
    and may change interfaces and not be backwards compatible between versions.

    In addition, this operation may not be supported by all hardware or third-party
    simulators.
    """

    def __init__(self, target: sympy.Symbol, expression: sympy.Basic) -> None:
        """Initializes the SetOperation object.

        Args:
             target: Symbol to modify the value of during execution.
             expression:  A SymPy expression to evaluate.  The value of which
                 will be assigned to the target symbol.
        """
        if not isinstance(target, sympy.Symbol):
            raise TypeError("Target must be a sympy.Symbol")
        if not isinstance(expression, sympy.Basic):
            expression = sympy.sympify(expression)

        self._target = target
        self._expression = expression

    def _value_equality_values_(self):
        return self._target, self._expression

    @property
    def target(self) -> sympy.Symbol:
        return self._target

    @property
    def expression(self) -> sympy.Basic:
        return self._expression

    @property
    def qubits(self) -> tuple[cirq.Qid, ...]:
        return ()

    def with_qubits(self, *qubits: cirq.Qid) -> SetVariable:
        if qubits:
            raise ValueError("SetVariable does not support qubits.")
        return self

    def _resolve_parameters_(self, resolver: cirq.ParamResolver, recursive: bool) -> SetVariable:
        resolved_expr = protocols.resolve_parameters(self._expression, resolver, recursive)
        return SetVariable(self._target, resolved_expr)

    def _is_parameterized_(self) -> bool:
        return any(a.free_symbols for a in self._expression.args)

    def _parameter_names_(self) -> Set[str]:
        return protocols.parameter_names(self._expression)

    def _act_on_(self, sim_state: cirq.SimulationStateBase) -> bool:
        replacements = {}
        for symbol in self._expression.free_symbols:
            if not isinstance(symbol, sympy.Symbol):
                continue
            name = symbol.name
            key = value.MeasurementKey.parse_serialized(name)

            # Check if it is a measurement key first
            if key in sim_state.classical_data.keys():
                replacements[symbol] = sim_state.classical_data.get_int(key)
            # Check if it is in the parameter resolver
            elif (
                symbol in sim_state.param_resolver.param_dict
                or name in sim_state.param_resolver.param_dict
            ):
                replacements[symbol] = sim_state.param_resolver.value_of(symbol)
            else:
                raise ValueError(
                    f"Symbol '{symbol}' in expression '{self._expression}' could not be resolved. "
                    "It must be present in the parameter resolver or as a measurement key."
                )

        # Handle Indexed symbols (e.g. bitwise measurements)
        for indexed in self._expression.atoms(sympy.Indexed):
            name = indexed.base.name
            key = value.MeasurementKey.parse_serialized(name)
            if key in sim_state.classical_data.keys():
                digits = sim_state.classical_data.get_digits(key)
                idx = int(indexed.indices[0])
                if 0 <= idx < len(digits):
                    replacements[indexed] = digits[idx]
                else:
                    raise ValueError(f"Index {idx} of symbol {key} out of bounds.")
            else:
                raise ValueError(f"Symbol {key} in expressions has no measurements.")

        evaluated_value = self._expression.subs(replacements)
        if evaluated_value.is_integer:
            evaluated_value = int(evaluated_value)
        elif evaluated_value.is_real:
            evaluated_value = float(evaluated_value)
        elif evaluated_value.is_number:
            evaluated_value = complex(evaluated_value)
        else:
            raise ValueError(
                f"Expression '{self._expression}' did not evaluate to a number: {evaluated_value}"
            )

        # Update the parameter resolver in sim_state
        current_params = dict(sim_state.param_resolver.param_dict)
        current_params[self._target] = evaluated_value
        sim_state.param_resolver = resolver.ParamResolver(current_params)
        return True

    def __repr__(self) -> str:
        import cirq._compat as _compat

        return (
            f"cirq.SetVariable({_compat.proper_repr(self._target)}, "
            f"{_compat.proper_repr(self._expression)})"
        )

    def __str__(self) -> str:
        return f"SetVariable({self._target}, {self._expression})"

    def _json_dict_(self) -> dict[str, Any]:
        return {'target': self._target, 'expression': self._expression}

    @classmethod
    def _from_json_dict_(cls, target, expression, **kwargs):
        return cls(target=target, expression=expression)
