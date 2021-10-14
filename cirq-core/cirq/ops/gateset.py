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


def _gate_str(
    gate: Union[raw_types.Gate, Type[raw_types.Gate], 'cirq.GateFamily'],
    gettr: Callable[[Any], str] = str,
) -> str:
    return gettr(gate) if not isinstance(gate, type) else f'{gate.__module__}.{gate.__name__}'


@value.value_equality(distinct_child_types=True)
class GateFamily:
    """Wrapper around gate instances/types describing a set of accepted gates.

    GateFamily supports initialization via
        a) Non-parameterized instances of `cirq.Gate` (Instance Family).
        b) Python types inheriting from `cirq.Gate` (Type Family).

    By default, the containment checks depend on the initialization type:
        a) Instance Family: Containment check is done via `cirq.equal_up_to_global_phase`.
        b) Type Family: Containment check is done by type comparison.

    For example:
        a) Instance Family:
            >>> gate_family = cirq.GateFamily(cirq.X)
            >>> assert cirq.X in gate_family
            >>> assert cirq.Rx(rads=np.pi) in gate_family
            >>> assert cirq.X ** sympy.Symbol("theta") not in gate_family

        b) Type Family:
            >>> gate_family = cirq.GateFamily(cirq.XPowGate)
            >>> assert cirq.X in gate_family
            >>> assert cirq.Rx(rads=np.pi) in gate_family
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
        ignore_global_phase: bool = True,
    ) -> None:
        """Init GateFamily.

        Args:
            gate: A python `type` inheriting from `cirq.Gate` for type based membership checks, or
                a non-parameterized instance of a `cirq.Gate` for equality based membership checks.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            ignore_global_phase: If True, value equality is checked via
                `cirq.equal_up_to_global_phase`.

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
        self._ignore_global_phase = ignore_global_phase

    def _gate_str(self, gettr: Callable[[Any], str] = str) -> str:
        return _gate_str(self.gate, gettr)

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
            a) Instance Family: `cirq.equal_up_to_global_phase(gate, self.gate)`
                                 if self._ignore_global_phase else `gate == self.gate`.
            b) Type Family: `isinstance(gate, self.gate)`.

        Args:
            gate: `cirq.Gate` instance which should be checked for containment.
        """
        if isinstance(self.gate, raw_types.Gate):
            return (
                protocols.equal_up_to_global_phase(gate, self.gate)
                if self._ignore_global_phase
                else gate == self._gate
            )
        return isinstance(gate, self.gate)

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        if isinstance(item, raw_types.Operation):
            if item.gate is None:
                return False
            item = item.gate
        return self._predicate(item)

    def __str__(self) -> str:
        return f'{self.name}\n{self.description}'

    def __repr__(self) -> str:
        name_and_description = ''
        if self.name != self._default_name() or self.description != self._default_description():
            name_and_description = f'name="{self.name}", description="{self.description}", '
        return (
            f'cirq.GateFamily('
            f'gate={self._gate_str(repr)}, '
            f'{name_and_description}'
            f'ignore_global_phase={self._ignore_global_phase})'
        )

    def _value_equality_values_(self) -> Any:
        # `isinstance` is used to ensure the a gate type and gate instance is not compared.
        return (
            isinstance(self.gate, raw_types.Gate),
            self.gate,
            self.name,
            self.description,
            self._ignore_global_phase,
        )


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
        accept_global_phase_op: bool = True,
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
            accept_global_phase_op: If True, `cirq.GlobalPhaseOperation` is accepted.
        """
        self._name = name
        self._unroll_circuit_op = unroll_circuit_op
        self._accept_global_phase_op = accept_global_phase_op
        self._instance_gate_families: Dict[raw_types.Gate, GateFamily] = {}
        self._type_gate_families: Dict[Type[raw_types.Gate], GateFamily] = {}
        self._gates_repr_str = ", ".join([_gate_str(g, repr) for g in gates])
        unique_gate_list: List[GateFamily] = list(
            dict.fromkeys(g if isinstance(g, GateFamily) else GateFamily(gate=g) for g in gates)
        )
        for g in unique_gate_list:
            if type(g) == GateFamily:
                if isinstance(g.gate, raw_types.Gate):
                    self._instance_gate_families[g.gate] = g
                else:
                    self._type_gate_families[g.gate] = g
        self._gates_str_str = "\n\n".join([str(g) for g in unique_gate_list])
        self._gates = frozenset(unique_gate_list)

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
        accept_global_phase_op: Optional[bool] = None,
    ) -> 'Gateset':
        """Returns a copy of this Gateset with identical gates and new values for named arguments.

        If a named argument is None then corresponding value of this Gateset is used instead.

        Args:
            name: New name for the Gateset.
            unroll_circuit_op: If True, new Gateset will recursively validate
                `cirq.CircuitOperation` by validating the underlying `cirq.Circuit`.
            accept_global_phase_op: If True, new Gateset will accept `cirq.GlobalPhaseOperation`.

        Returns:
            `self` if all new values are None or identical to the values of current Gateset.
            else a new Gateset with identical gates and new values for named arguments.
        """

        def val_if_none(var: Any, val: Any) -> Any:
            return var if var is not None else val

        name = val_if_none(name, self._name)
        unroll_circuit_op = val_if_none(unroll_circuit_op, self._unroll_circuit_op)
        accept_global_phase_op = val_if_none(accept_global_phase_op, self._accept_global_phase_op)
        if (
            name == self._name
            and unroll_circuit_op == self._unroll_circuit_op
            and accept_global_phase_op == self._accept_global_phase_op
        ):
            return self
        return Gateset(
            *self.gates,
            name=name,
            unroll_circuit_op=cast(bool, unroll_circuit_op),
            accept_global_phase_op=cast(bool, accept_global_phase_op),
        )

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        """Check for containment of a given Gate/Operation in this Gateset.

        Containment checks are handled as follows:
            a) For Gates or Operations that have an underlying gate (i.e. op.gate is not None):
                - Forwards the containment check to the underlying `cirq.GateFamily` objects.
                - Examples of such operations include `cirq.GateOperations` and their controlled
                    and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                    `cirq.ControlledOperation` where `op.gate` is not None) etc.

            b) For Operations that do not have an underlying gate:
                - Forwards the containment check to `self._validate_operation(item)`.
                - Examples of such operations include `cirq.CircuitOperations` and their controlled
                    and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                    `cirq.ControlledOperation` where `op.gate` is None) etc.

        The complexity of the method is:
            a) O(1) when any default `cirq.GateFamily` instance accepts the given item, except
                for an Instance GateFamily trying to match an item with a different global phase.
            b) O(n) for all other cases: matching against custom gate families, matching across
                global phase for the default Instance GateFamily, no match against any underlying
                gate family.

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

        return any(item in gate_family for gate_family in self._gates)

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
            return self._accept_global_phase_op
        else:
            return False

    def _value_equality_values_(self) -> Any:
        return (
            self.gates,
            self.name,
            self._unroll_circuit_op,
            self._accept_global_phase_op,
        )

    def __repr__(self) -> str:
        name_str = f'name = "{self.name}", ' if self.name is not None else ''
        return (
            f'cirq.Gateset('
            f'{self._gates_repr_str}, '
            f'{name_str}'
            f'unroll_circuit_op = {self._unroll_circuit_op},'
            f'accept_global_phase_op = {self._accept_global_phase_op})'
        )

    def __str__(self) -> str:
        header = 'Gateset: '
        if self.name:
            header += self.name
        return f'{header}\n' + self._gates_str_str
