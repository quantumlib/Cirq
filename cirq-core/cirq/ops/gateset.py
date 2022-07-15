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

"""Functionality for grouping and validating Cirq gates."""

from typing import (
    Any,
    Callable,
    cast,
    Dict,
    FrozenSet,
    Hashable,
    List,
    Optional,
    Sequence,
    Type,
    TYPE_CHECKING,
    Union,
)

from cirq import protocols, value
from cirq.ops import global_phase_op, op_tree, raw_types

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

    - Non-parameterized instances of `cirq.Gate` (Instance Family).
    - Python types inheriting from `cirq.Gate` (Type Family).

    By default, the containment checks depend on the initialization type:

    - Instance Family: Containment check is done via `cirq.equal_up_to_global_phase`.
    - Type Family: Containment check is done by type comparison.

    For example:

    - Instance Family:

        >>> gate_family = cirq.GateFamily(cirq.X)
        >>> assert cirq.X in gate_family
        >>> assert cirq.Rx(rads=np.pi) in gate_family
        >>> assert cirq.X ** sympy.Symbol("theta") not in gate_family

    - Type Family:

        >>> gate_family = cirq.GateFamily(cirq.XPowGate)
        >>> assert cirq.X in gate_family
        >>> assert cirq.Rx(rads=np.pi) in gate_family
        >>> assert cirq.X ** sympy.Symbol("theta") in gate_family

    As seen in the examples above, GateFamily supports containment checks for instances of both
    `cirq.Operation` and `cirq.Gate`. By default, a `cirq.Operation` instance `op` is accepted if
    the underlying `op.gate` is accepted.

    Further constraints can be added on containment checks for `cirq.Operation` objects by setting
    `tags_to_accept` and/or `tags_to_ignore` in the GateFamily constructor. For a tagged
    operation, the underlying gate `op.gate` will be checked for containment only if both:

    - `op.tags` has no intersection with `tags_to_ignore`
    - `tags_to_accept` is not empty, then `op.tags` should have a non-empty intersection with
        `tags_to_accept`.

    If a `cirq.Operation` contains tags from both `tags_to_accept` and `tags_to_ignore`, it is
    rejected. Furthermore, tags cannot appear in both `tags_to_accept` and `tags_to_ignore`.

    For the purpose of tag comparisons, a `Gate` is considered as an `Operation` without tags.

    For example:

        >>> q = cirq.NamedQubit('q')
        >>> gate_family = cirq.GateFamily(cirq.ZPowGate, tags_to_accept=['accepted_tag'])
        >>> assert cirq.Z(q).with_tags('accepted_tag') in gate_family
        >>> assert cirq.Z(q).with_tags('other_tag') not in gate_family
        >>> assert cirq.Z(q) not in gate_family
        >>> assert cirq.Z not in gate_family
        ...
        >>> gate_family = cirq.GateFamily(cirq.ZPowGate, tags_to_ignore=['ignored_tag'])
        >>> assert cirq.Z(q).with_tags('ignored_tag') not in gate_family
        >>> assert cirq.Z(q).with_tags('other_tag') in gate_family
        >>> assert cirq.Z(q) in gate_family
        >>> assert cirq.Z in gate_family

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
        tags_to_accept: Sequence[Hashable] = (),
        tags_to_ignore: Sequence[Hashable] = (),
    ) -> None:
        """Init GateFamily.

        Args:
            gate: A python `type` inheriting from `cirq.Gate` for type based membership checks, or
                a non-parameterized instance of a `cirq.Gate` for equality based membership checks.
            name: The name of the gate family.
            description: Human readable description of the gate family.
            ignore_global_phase: If True, value equality is checked via
                `cirq.equal_up_to_global_phase`.
            tags_to_accept: If non-empty, only `cirq.Operations` containing at least one tag in this
                sequence can be accepted.
            tags_to_ignore: Any `cirq.Operation` containing at least one tag in this sequence is
                rejected. Note that this takes precedence over `tags_to_accept`, so an operation
                which contains tags from both `tags_to_accept` and `tags_to_ignore` is rejected.

        Raises:
            ValueError: if `gate` is not a `cirq.Gate` instance or subclass.
            ValueError: if `gate` is a parameterized instance of `cirq.Gate`.
            ValueError: if `tags_to_accept` and `tags_to_ignore` contain common tags.
        """
        if not (
            isinstance(gate, raw_types.Gate)
            or (isinstance(gate, type) and issubclass(gate, raw_types.Gate))
        ):
            raise ValueError(f'Gate {gate} must be an instance or subclass of `cirq.Gate`.')
        if isinstance(gate, raw_types.Gate) and protocols.is_parameterized(gate):
            raise ValueError(f'Gate {gate} must be a non-parameterized instance of `cirq.Gate`.')

        self._gate = gate
        self._tags_to_accept = frozenset(tags_to_accept)
        self._tags_to_ignore = frozenset(tags_to_ignore)
        self._name = name if name else self._default_name()
        self._description = description if description else self._default_description()
        self._ignore_global_phase = ignore_global_phase

        common_tags = self._tags_to_accept & self._tags_to_ignore
        if common_tags:
            raise ValueError(
                f"Tag(s) '{list(common_tags)}' cannot be in both tags_to_accept and tags_to_ignore."
            )

    def _gate_str(self, gettr: Callable[[Any], str] = str) -> str:
        return _gate_str(self.gate, gettr)

    def _gate_json(self) -> Union[raw_types.Gate, str]:
        return self.gate if not isinstance(self.gate, type) else protocols.json_cirq_type(self.gate)

    def _default_name(self) -> str:
        family_type = 'Instance' if isinstance(self.gate, raw_types.Gate) else 'Type'
        return f'{family_type} GateFamily: {self._gate_str()}'

    def _default_description(self) -> str:
        check_type = r'g == {}' if isinstance(self.gate, raw_types.Gate) else r'isinstance(g, {})'
        tags_to_accept_str = (
            f'\nAccepted tags: {list(self._tags_to_accept)}' if self._tags_to_accept else ''
        )
        tags_to_ignore_str = (
            f'\nIgnored tags: {list(self._tags_to_ignore)}' if self._tags_to_ignore else ''
        )
        return (
            f'Accepts `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}`'
            + tags_to_accept_str
            + tags_to_ignore_str
        )

    @property
    def gate(self) -> Union[Type[raw_types.Gate], raw_types.Gate]:
        return self._gate

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def tags_to_accept(self) -> FrozenSet[Hashable]:
        return self._tags_to_accept

    @property
    def tags_to_ignore(self) -> FrozenSet[Hashable]:
        return self._tags_to_ignore

    def _predicate(self, gate: raw_types.Gate) -> bool:
        """Checks whether `cirq.Gate` instance `gate` belongs to this GateFamily.

        The default predicate depends on the gate family initialization type:

        - Instance Family: `cirq.equal_up_to_global_phase(gate, self.gate)`
            if self._ignore_global_phase else `gate == self.gate`.
        - Type Family: `isinstance(gate, self.gate)`.

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
        if self._tags_to_accept and (
            not isinstance(item, raw_types.Operation) or self._tags_to_accept.isdisjoint(item.tags)
        ):
            return False
        if isinstance(item, raw_types.Operation) and not self._tags_to_ignore.isdisjoint(item.tags):
            return False

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
            f'ignore_global_phase={self._ignore_global_phase}, '
            f'tags_to_accept={self._tags_to_accept}, '
            f'tags_to_ignore={self._tags_to_ignore})'
        )

    def _value_equality_values_(self) -> Any:
        # `isinstance` is used to ensure the a gate type and gate instance is not compared.
        return (
            isinstance(self.gate, raw_types.Gate),
            self.gate,
            self.name,
            self.description,
            self._ignore_global_phase,
            self._tags_to_accept,
            self._tags_to_ignore,
        )

    def _json_dict_(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            'gate': self._gate_json(),
            'name': self.name,
            'description': self.description,
            'ignore_global_phase': self._ignore_global_phase,
        }
        if self._tags_to_accept:
            d['tags_to_accept'] = list(self._tags_to_accept)
        if self._tags_to_ignore:
            d['tags_to_ignore'] = list(self._tags_to_ignore)
        return d

    @classmethod
    def _from_json_dict_(
        cls,
        gate,
        name,
        description,
        ignore_global_phase,
        tags_to_accept=(),
        tags_to_ignore=(),
        **kwargs,
    ) -> 'GateFamily':
        if isinstance(gate, str):
            gate = protocols.cirq_type_from_json(gate)
        return cls(
            gate,
            name=name,
            description=description,
            ignore_global_phase=ignore_global_phase,
            tags_to_accept=tags_to_accept,
            tags_to_ignore=tags_to_ignore,
        )


@value.value_equality()
class Gateset:
    """Gatesets represent a collection of `cirq.GateFamily` objects.

    Gatesets are useful for

    - Describing the set of allowed gates in a human-readable format.
    - Validating a given gate / `cirq.OP_TREE` against the set of allowed gates.

    Gatesets rely on the underlying `cirq.GateFamily` for both description and
    validation purposes.
    """

    def __init__(
        self,
        *gates: Union[Type[raw_types.Gate], raw_types.Gate, GateFamily],
        name: Optional[str] = None,
        unroll_circuit_op: bool = True,
    ) -> None:
        """Init Gateset.

        Accepts a list of gates, each of which should be either

        - `cirq.Gate` subclass
        - `cirq.Gate` instance
        - `cirq.GateFamily` instance

        `cirq.Gate` subclasses and instances are converted to the default
        `cirq.GateFamily(gate=g)` instance and thus a default name and
        description is populated.

        Args:
            *gates: A list of `cirq.Gate` subclasses / `cirq.Gate` instances /
                `cirq.GateFamily` instances to initialize the Gateset.
            name: (Optional) Name for the Gateset. Useful for description.
            unroll_circuit_op: If True, `cirq.CircuitOperation` is recursively
                validated by validating the underlying `cirq.Circuit`.
        """
        self._name = name
        self._unroll_circuit_op = unroll_circuit_op
        self._instance_gate_families: Dict[raw_types.Gate, GateFamily] = {}
        self._type_gate_families: Dict[Type[raw_types.Gate], GateFamily] = {}
        self._gates_repr_str = ", ".join([_gate_str(g, repr) for g in gates])
        unique_gate_list: List[GateFamily] = list(
            dict.fromkeys(g if isinstance(g, GateFamily) else GateFamily(gate=g) for g in gates)
        )

        for g in unique_gate_list:
            if type(g) is GateFamily and not (g.tags_to_ignore or g.tags_to_accept):
                if isinstance(g.gate, raw_types.Gate):
                    self._instance_gate_families[g.gate] = g
                else:
                    self._type_gate_families[g.gate] = g
        self._unique_gate_list = unique_gate_list
        self._gates = frozenset(unique_gate_list)

    @property
    def name(self) -> Optional[str]:
        return self._name

    @property
    def gates(self) -> FrozenSet[GateFamily]:
        return self._gates

    def with_params(
        self, *, name: Optional[str] = None, unroll_circuit_op: Optional[bool] = None
    ) -> 'Gateset':
        """Returns a copy of this Gateset with identical gates and new values for named arguments.

        If a named argument is None then corresponding value of this Gateset is used instead.

        Args:
            name: New name for the Gateset.
            unroll_circuit_op: If True, new Gateset will recursively validate
                `cirq.CircuitOperation` by validating the underlying `cirq.Circuit`.

        Returns:
            `self` if all new values are None or identical to the values of current Gateset.
            else a new Gateset with identical gates and new values for named arguments.
        """

        def val_if_none(var: Any, val: Any) -> Any:
            return var if var is not None else val

        name = val_if_none(name, self._name)
        unroll_circuit_op = val_if_none(unroll_circuit_op, self._unroll_circuit_op)
        if name == self._name and unroll_circuit_op == self._unroll_circuit_op:
            return self
        gates = self.gates
        return Gateset(*gates, name=name, unroll_circuit_op=cast(bool, unroll_circuit_op))

    def __contains__(self, item: Union[raw_types.Gate, raw_types.Operation]) -> bool:
        r"""Check for containment of a given Gate/Operation in this Gateset.

        Containment checks are handled as follows:

        - For Gates or Operations that have an underlying gate (i.e. op.gate is not None):
            - Forwards the containment check to the underlying `cirq.GateFamily` objects.
            - Examples of such operations include `cirq.GateOperation`s and their controlled
                and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                `cirq.ControlledOperation` where `op.gate` is not None) etc.
        - For Operations that do not have an underlying gate:
            - Forwards the containment check to `self._validate_operation(item)`.
            - Examples of such operations include `cirq.CircuitOperation`s and their controlled
                and tagged variants (i.e. instances of `cirq.TaggedOperation`,
                `cirq.ControlledOperation` where `op.gate` is None) etc.

        The complexity of the method in terms of the number of `gates`, n, is

        - O(1) when any default `cirq.GateFamily` instance accepts the given item, except
            for an Instance GateFamily trying to match an item with a different global phase.
        - O(n) for all other cases: matching against custom gate families, matching across
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

    def validate(self, circuit_or_optree: Union['cirq.AbstractCircuit', op_tree.OP_TREE]) -> bool:
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

        - For any operation which has an underlying gate (i.e. `op.gate` is not None):
            - Containment is checked via `self.__contains__` which further checks for containment
                in any of the underlying gate families.
        - For all other types of operations (eg: `cirq.CircuitOperation`,
            etc):
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
            return self.validate(op.mapped_circuit(deep=True))
        else:
            return False

    def _value_equality_values_(self) -> Any:
        return (self.gates, self.name, self._unroll_circuit_op)

    def __repr__(self) -> str:
        name_str = f'name = "{self.name}", ' if self.name is not None else ''
        gates_str = f'{self._gates_repr_str}, ' if len(self._gates_repr_str) > 0 else ''
        return (
            f'cirq.Gateset('
            f'{gates_str}'
            f'{name_str}'
            f'unroll_circuit_op = {self._unroll_circuit_op})'
        )

    def __str__(self) -> str:
        header = 'Gateset: '
        if self.name:
            header += self.name
        return f'{header}\n' + "\n\n".join([str(g) for g in self._unique_gate_list])

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'gates': self._unique_gate_list,
            'name': self.name,
            'unroll_circuit_op': self._unroll_circuit_op,
        }

    @classmethod
    def _from_json_dict_(cls, gates, name, unroll_circuit_op, **kwargs) -> 'Gateset':
        # This parameter was deprecated in 0.16, but we keep this logic here for backwards
        # compatibility.
        if 'accept_global_phase_op' in kwargs:
            accept_global_phase_op = kwargs['accept_global_phase_op']
            global_phase_family = GateFamily(gate=global_phase_op.GlobalPhaseGate)
            if accept_global_phase_op is True:
                gates.append(global_phase_family)
            elif accept_global_phase_op is False:
                gates = [
                    family for family in gates if family.gate is not global_phase_op.GlobalPhaseGate
                ]
        return cls(*gates, name=name, unroll_circuit_op=unroll_circuit_op)
