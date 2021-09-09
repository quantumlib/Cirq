# Copyright 2021 The Cirq Developers
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

"""Types defining GateFamily and Gateset"""

from typing import Any, Callable, Dict, FrozenSet, Optional, Type, TYPE_CHECKING, Union
from cirq.ops import global_phase_op, op_tree, raw_types
from cirq import protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality(distinct_child_types=True)
class GateFamily:
    """Wrapper around gate instances/types describing a set of accepted gates.

    By default, GateFamily supports initialization and containment for
        a) Non-parameterized `cirq.Gate` instances.
        b) Gate types deriving from `cirq.Gate`.

    In order to create gate families with constraints on parameters of a gate
    type, users should derive from the GateFamily class and override the
    predicate method used to check for containment.
    """

    def __init__(
        self,
        gate: Union[Type[raw_types.Gate], raw_types.Gate],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Init GateFamily
        Args:
            gate: A gate type or a non-parameterized instance of a gate type.
            name: The name of the gate family.
            description: Human readable description of the gate family.

        Raises:
            ValueError: if `gate` is not a `cirq.Gate` instance or subclass.
        """
        if not (
            isinstance(gate, raw_types.Gate)
            or (isinstance(gate, type) and issubclass(gate, raw_types.Gate))
        ):
            raise ValueError(f'Gate {gate} must be an instance or subclass of `cirq.Gate`.')
        if isinstance(gate, raw_types.Gate) and protocols.is_parameterized(gate):
            raise ValueError(f'Gate {gate} must be a non-parameterized instance of `cirq.Gate`.')

        self._gate = gate
        self._name = name if name else self._default_name()
        self._description = description if description else self._default_description()

    def _gate_str(self, gettr: Callable[[Any], str] = str) -> str:
        return (
            gettr(self.gate)
            if isinstance(self.gate, raw_types.Gate)
            else f'{self.gate.__module__}.{self.gate.__name__}'
        )

    def _default_name(self) -> str:
        if isinstance(self.gate, raw_types.Gate):
            return f'Instance GateFamily: {self._gate_str()}'
        else:
            return f'Type GateFamily: {self._gate_str()}'

    def _default_description(self) -> str:
        if isinstance(self.gate, raw_types.Gate):
            return f'Accepts `cirq.Gate` instances `g` s.t. `g == {self._gate_str()}`.'
        else:
            return f'Accepts `cirq.Gate` instances `g` s.t. `isinstance(g, {self._gate_str()})`.'

    @property
    def gate(self) -> Union[Type[raw_types.Gate], raw_types.Gate]:
        return self._gate

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def predicate(self, g: raw_types.Gate) -> bool:
        """Checks whether gate instance `g` belongs to this GateFamily."""
        return g == self.gate if isinstance(self.gate, raw_types.Gate) else isinstance(g, self.gate)

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if isinstance(item, raw_types.Operation):
            if item.gate is None:
                return False
            item = item.gate
        return self.predicate(item)

    def __str__(self) -> str:
        return f'{self.name}\n{self.description}'

    def __repr__(self) -> str:
        return (
            f'cirq.GateFamily(gate={self._gate_str(repr)},'
            f'name="{self.name}", '
            f'description="{self.description}")'
        )

    def _value_equality_values_(self) -> Any:
        # `isinstance` is used to ensure the a gate type and gate instance is not compared.
        return isinstance(self.gate, raw_types.Gate), self.gate, self.name, self.description


@value.value_equality()
class Gateset:
    """Gatesets represent a collection of GateFamily objects.

    Gatesets are useful for
        a) Describing the set of allowed gates in a human readable format
        b) Validating a given gate / optree against the set of allowed gates

    Gatesets rely on the underlying GateFamily's for both description and
    validation purposes.

    The default GateFamily can be used to description and validation of the
    two most common cases:
        a) A specific gate instance is allowed
        b) All instances of a specific gate class are allowed.

    Users can define custom gate families for more complex scenarios and
    use the GateFamily objects as members of a Gateset.
    """

    def __init__(
        self,
        *gates: Union[Type[raw_types.Gate], raw_types.Gate, GateFamily],
        name: Optional[str] = None,
    ) -> None:
        """Inits Gateset.

        Accepts a list of gates, each of which should be either
            a) `cirq.Gate` subclass
            b) `cirq.Gate` instance
            c) `cirq.GateFamily` instance

        `cirq.Gate` subclasses and instances are converted to the default
        `cirq.GateFamily(gate=g)` instance and thus a default name and
        description is populated.

        Args:
            gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
            `cirq.GateFamily` instances to initialize the Gateset.
            name: (Optional) Name for the Gateset. Useful for description.
        """
        self._name = name
        self._gates = frozenset(
            [g if isinstance(g, GateFamily) else GateFamily(gate=g) for g in gates]
        )

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def gates(self) -> FrozenSet[GateFamily]:
        return self._gates

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        """Check for containment of `item` in any of the member `GateFamily`.

        This method just forwards containment checks to the underlying
        `GateFamily` objects. It returns `True` if any of the member gate
        families can accept `item`.

        To validate whether a `cirq.CircuitOperation`/`cirq.OP_TREE`/
        `cirq.Circuit` is made up entirely of gates from this Gateset, use the
        `validate` method given below.

        Args:
            item: The `cirq.Gate` or `cirq.Operation` instance to check
                containment for.
        """
        for gate_family in self.gates:
            if item in gate_family:
                return True
        return False

    def validate(
        self,
        circuit_or_optree: Union['cirq.AbstractCircuit', op_tree.OP_TREE],
        *,
        unroll_circuit_op: bool = True,
        accept_global_phase: bool = True,
    ) -> bool:
        """Validates gates forming `circuit_or_optree` should be contained in Gateset.

        Args:
            circuit_or_optree: The `cirq.Circuit` or `cirq.OP_TREE` to validate.
            unroll_circuit_op: If True, `cirq.CircuitOperation` is recursively
                validated by validating the underlying `cirq.Circuit`.
            accept_global_phase: If True, `cirq.GlobalPhaseOperation` is accepted.
        """
        # To avoid circular import.
        from cirq.circuits import circuit

        optree = circuit_or_optree
        if isinstance(circuit_or_optree, circuit.AbstractCircuit):
            optree = circuit_or_optree.all_operations()
        return all(
            self._validate_operation(
                op, unroll_circuit_op=unroll_circuit_op, accept_global_phase=accept_global_phase
            )
            for op in op_tree.flatten_to_ops(optree)
        )

    def _validate_operation(
        self,
        op: raw_types.Operation,
        *,
        unroll_circuit_op: bool = True,
        accept_global_phase: bool = True,
    ) -> bool:
        # To avoid circular import.
        from cirq.circuits import circuit_operation

        if isinstance(op, raw_types.TaggedOperation):
            return self._validate_operation(
                op.sub_operation,
                unroll_circuit_op=unroll_circuit_op,
                accept_global_phase=accept_global_phase,
            )
        elif isinstance(op, circuit_operation.CircuitOperation) and unroll_circuit_op:
            return self.validate(
                op.mapped_circuit(deep=True),
                unroll_circuit_op=unroll_circuit_op,
                accept_global_phase=accept_global_phase,
            )
        elif isinstance(op, global_phase_op.GlobalPhaseOperation):
            return accept_global_phase
        elif op.gate is not None:
            return op in self
        else:
            return False

    def _value_equality_values_(self) -> Any:
        return frozenset(self.gates), self.name

    def __repr__(self) -> str:
        return f'cirq.Gateset({",".join([repr(g) for g in self.gates])}, name = "{self.name}")'

    def __str__(self) -> str:
        header = 'Gateset: '
        if self.name:
            header += self.name
        return f'{header}\n' + "\n\n".join([str(g) for g in self.gates])
