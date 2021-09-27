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

"""Functionality for grouping and validating Cirq Gates"""

from typing import Any, Callable, cast, Dict, FrozenSet, List, Optional, Type, TYPE_CHECKING, Union
from cirq.ops import global_phase_op, op_tree, raw_types
from cirq import protocols, value

if TYPE_CHECKING:
    import cirq


@value.value_equality(distinct_child_types=True)
class GateFamily:
    """Wrapper around gate instances/types describing a set of accepted gates.

    GateFamily supports initialization via
        a) Non-parameterized instances of `cirq.Gate` (Instance Family).
        b) Python types inheriting from `cirq.Gate` (Type Family).

    By default, the containment checks depend on the initialization type:
        a) Instance Family: Containment check is done by object equality.
        b) Type Family: Containment check is done by type comparison.

    For example:
        a) Instance Family:
            >>> gate_family = cirq.GateFamily(cirq.X)
            >>> assert cirq.X in gate_family
            >>> assert cirq.X ** sympy.Symbol("theta") not in gate_family

        b) Type Family:
            >>> gate_family = cirq.GateFamily(cirq.XPowGate)
            >>> assert cirq.X in gate_family
            >>> assert cirq.X ** sympy.Symbol("theta") in gate_family

    In order to create gate families with constraints on parameters of a gate
    type, users should derive from the `cirq.GateFamily` class and override the
    `_predicate` method used to check for gate containment.
    """

    def __init__(
        self,
        gate: Union[Type[raw_types.Gate], raw_types.Gate],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> None:
        """Init GateFamily.

        Args:
            gate: A python `type` inheriting from `cirq.Gate` for type based membership checks, or
                a non-parameterized instance of a `cirq.Gate` for equality based membership checks.
            name: The name of the gate family.
            description: Human readable description of the gate family.

        Raises:
            ValueError: if `gate` is not a `cirq.Gate` instance or subclass.
            ValueError: if `gate` is a parameterized instance of `cirq.Gate`.
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
        family_type = 'Instance' if isinstance(self.gate, raw_types.Gate) else 'Type'
        return f'{family_type} GateFamily: {self._gate_str()}'

    def _default_description(self) -> str:
        check_type = r'g == {}' if isinstance(self.gate, raw_types.Gate) else r'isinstance(g, {})'
        return f'Accepts `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}`'

    @property
    def gate(self) -> Union[Type[raw_types.Gate], raw_types.Gate]:
        return self._gate

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    def _predicate(self, gate: raw_types.Gate) -> bool:
        """Checks whether `cirq.Gate` instance `gate` belongs to this GateFamily.

        The default predicate depends on the gate family initialization type:
            a) Instance Family: `gate == self.gate`.
            b) Type Family: `isinstance(gate, self.gate)`.

        Args:
            gate: `cirq.Gate` instance which should be checked for containment.
        """
        return (
            gate == self.gate
            if isinstance(self.gate, raw_types.Gate)
            else isinstance(gate, self.gate)
        )

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if isinstance(item, raw_types.Operation):
            if item.gate is None:
                return False
            item = item.gate
        return self._predicate(item)

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
    """Gatesets represent a collection of `cirq.GateFamily` objects.

    Gatesets are useful for
        a) Describing the set of allowed gates in a human readable format
        b) Validating a given gate / optree against the set of allowed gates

    Gatesets rely on the underlying `cirq.GateFamily` for both description and
    validation purposes.
    """

    def __init__(
        self,
        *gates: Union[Type[raw_types.Gate], raw_types.Gate, GateFamily],
        name: Optional[str] = None,
        unroll_circuit_op: bool = True,
        accept_global_phase: bool = True,
    ) -> None:
        """Init Gateset.

        Accepts a list of gates, each of which should be either
            a) `cirq.Gate` subclass
            b) `cirq.Gate` instance
            c) `cirq.GateFamily` instance

        `cirq.Gate` subclasses and instances are converted to the default
        `cirq.GateFamily(gate=g)` instance and thus a default name and
        description is populated.

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
            `cirq.GateFamily` instances to initialize the Gateset.
            name: (Optional) Name for the Gateset. Useful for description.
            unroll_circuit_op: If True, `cirq.CircuitOperation` is recursively
                validated by validating the underlying `cirq.Circuit`.
            accept_global_phase: If True, `cirq.GlobalPhaseOperation` is accepted.
        """
        self._name = name
        self._gates = frozenset(
            g if isinstance(g, GateFamily) else GateFamily(gate=g) for g in gates
        )
        self._unroll_circuit_op = unroll_circuit_op
        self._accept_global_phase = accept_global_phase
        self._instance_gate_families: Dict[raw_types.Gate, GateFamily] = {}
        self._type_gate_families: Dict[Type[raw_types.Gate], GateFamily] = {}
        self._custom_gate_families: List[GateFamily] = []
        for g in self._gates:
            if type(g) == GateFamily:
                if isinstance(g.gate, raw_types.Gate):
                    self._instance_gate_families[g.gate] = g
                else:
                    self._type_gate_families[g.gate] = g
            else:
                self._custom_gate_families.append(g)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def gates(self) -> FrozenSet[GateFamily]:
        return self._gates

    def with_params(
        self,
        *,
        name: Optional[str] = None,
        unroll_circuit_op: Optional[bool] = None,
        accept_global_phase: Optional[bool] = None,
    ) -> 'Gateset':
        """Returns a copy of this Gateset with identical gates and new values for named arguments.

        If a named argument is None then corresponding value of this Gateset is used instead.

        Args:
            name: New name for the Gateset.
            unroll_circuit_op: If True, new Gateset will recursively validate
                `cirq.CircuitOperation` by validating the underlying `cirq.Circuit`.
            accept_global_phase: If True, new Gateset will accept `cirq.GlobalPhaseOperation`.

        Returns:
            `self` if all new values are None or identical to the values of current Gateset.
            else a new Gateset with identical gates and new values for named arguments.
        """

        def val_if_none(var: Any, val: Any) -> Any:
            return var if var is not None else val

        name = val_if_none(name, self._name)
        unroll_circuit_op = val_if_none(unroll_circuit_op, self._unroll_circuit_op)
        accept_global_phase = val_if_none(accept_global_phase, self._accept_global_phase)
        if (
            name == self._name
            and unroll_circuit_op == self._unroll_circuit_op
            and accept_global_phase == self._accept_global_phase
        ):
            return self
        return Gateset(
            *self.gates,
            name=name,
            unroll_circuit_op=cast(bool, unroll_circuit_op),
            accept_global_phase=cast(bool, accept_global_phase),
        )

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        """Check for containment of a given Gate/Operation in this Gateset.

        Containment checks are handled as follows:
            a) For Gates or Operations that have an underlying gate (i.e. op.gate is not None):
                - Forwards the containment check to the underlying GateFamily's
                - Examples of such operations include `cirq.GateOperations` and their controlled
                    and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                    `cirq.ControlledOperation` where `op.gate` is not None) etc.

            b) For Operations that do not have an underlying gate:
                - Forwards the containment check to `self._validate_operation(item)`.
                - Examples of such operations include `cirq.CircuitOperations` and their controlled
                    and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                    `cirq.ControlledOperation` where `op.gate` is None) etc.

        The complexity of the method is:
            a) O(1) for checking containment in the default `cirq.GateFamily` instances.
            b) O(n) for checking containment in custom GateFamily instances.

        Args:
            item: The `cirq.Gate` or `cirq.Operation` instance to check containment for.
        """
        if isinstance(item, raw_types.Operation) and item.gate is None:
            return self._validate_operation(item)

        g = item if isinstance(item, raw_types.Gate) else item.gate
        assert g is not None, f'`item`: {item} must be a gate or have a valid `item.gate`'

        if g in self._instance_gate_families:
            assert item in self._instance_gate_families[g], (
                f"{item} instance matches {self._instance_gate_families[g]} but "
                f"is not accepted by it."
            )
            return True

        for gate_mro_type in type(g).mro():
            if gate_mro_type in self._type_gate_families:
                assert item in self._type_gate_families[gate_mro_type], (
                    f"{g} type {gate_mro_type} matches Type GateFamily:"
                    f"{self._type_gate_families[gate_mro_type]} but is not accepted by it."
                )
                return True

        return any(item in gate_family for gate_family in self._custom_gate_families)

    def validate(
        self,
        circuit_or_optree: Union['cirq.AbstractCircuit', op_tree.OP_TREE],
    ) -> bool:
        """Validates gates forming `circuit_or_optree` should be contained in Gateset.

        Args:
            circuit_or_optree: The `cirq.Circuit` or `cirq.OP_TREE` to validate.
        """
        # To avoid circular import.
        from cirq.circuits import circuit

        optree = circuit_or_optree
        if isinstance(circuit_or_optree, circuit.AbstractCircuit):
            optree = circuit_or_optree.all_operations()
        return all(self._validate_operation(op) for op in op_tree.flatten_to_ops(optree))

    def _validate_operation(self, op: raw_types.Operation) -> bool:
        """Validates whether the given `cirq.Operation` is contained in this Gateset.

        The containment checks are handled as follows:

        a) For any operation which has an underlying gate (i.e. `op.gate` is not None):
            - Containment is checked via `self.__contains__` which further checks for containment
                in any of the underlying gate families.

        b) For all other types of operations (eg: `cirq.CircuitOperation`,
        `cirq.GlobalPhaseOperation` etc):
            - The behavior is controlled via flags passed to the constructor.

        Users should override this method to define custom behavior for operations that do not
        have an underlying `cirq.Gate`.

        Args:
            op: The `cirq.Operation` instance to check containment for.
        """

        # To avoid circular import.
        from cirq.circuits import circuit_operation

        if op.gate is not None:
            return op in self

        if isinstance(op, raw_types.TaggedOperation):
            return self._validate_operation(op.sub_operation)
        elif isinstance(op, circuit_operation.CircuitOperation) and self._unroll_circuit_op:
            op_circuit = protocols.resolve_parameters(
                op.circuit.unfreeze(), op.param_resolver, recursive=False
            )
            op_circuit = op_circuit.transform_qubits(
                lambda q: cast(circuit_operation.CircuitOperation, op).qubit_map.get(q, q)
            )
            return self.validate(op_circuit)
        elif isinstance(op, global_phase_op.GlobalPhaseOperation):
            return self._accept_global_phase
        else:
            return False

    def _value_equality_values_(self) -> Any:
        return (
            frozenset(self.gates),
            self.name,
            self._unroll_circuit_op,
            self._accept_global_phase,
        )

    def __repr__(self) -> str:
        return (
            f'cirq.Gateset('
            f'{",".join([repr(g) for g in self.gates])},'
            f'name = "{self.name}",'
            f'unroll_circuit_op = {self._unroll_circuit_op},'
            f'accept_global_phase = {self._accept_global_phase})'
        )

    def __str__(self) -> str:
        header = 'Gateset: '
        if self.name:
            header += self.name
        return f'{header}\n' + "\n\n".join([str(g) for g in self.gates])
