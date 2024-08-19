# Copyright 2018 The Cirq Developers
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

"""Basic types defining qubits, gates, and operations."""

import abc
import functools
from typing import (
    cast,
    AbstractSet,
    Any,
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)
from typing_extensions import Self

import numpy as np
import sympy

from cirq import protocols, value
from cirq._import import LazyLoader
from cirq._compat import __cirq_debug__, _method_cache_name, cached_method
from cirq.type_workarounds import NotImplementedType
from cirq.ops import control_values as cv

# Lazy imports to break circular dependencies.
ops = LazyLoader("ops", globals(), "cirq.ops")
line_qubit = LazyLoader("line_qubit", globals(), "cirq.devices.line_qubit")


if TYPE_CHECKING:
    import cirq


class Qid(metaclass=abc.ABCMeta):
    """Identifies a quantum object such as a qubit, qudit, resonator, etc.

    Child classes represent specific types of objects, such as a qubit at a
    particular location on a chip or a qubit with a particular name.

    The main criteria that a custom qid must satisfy is *comparability*. Child
    classes meet this criteria by implementing the `_comparison_key` method. For
    example, `cirq.LineQubit`'s `_comparison_key` method returns `self.x`. This
    ensures that line qubits with the same `x` are equal, and that line qubits
    will be sorted ascending by `x`. `Qid` implements all equality,
    comparison, and hashing methods via `_comparison_key`.
    """

    @abc.abstractmethod
    def _comparison_key(self) -> Any:
        """Returns a value used to sort and compare this qubit with others.

        By default, qubits of differing type are sorted ascending according to
        their type name. Qubits of the same type are then sorted using their
        comparison key.
        """

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Returns the dimension or the number of quantum levels this qid has.
        E.g. 2 for a qubit, 3 for a qutrit, etc.
        """

    @staticmethod
    def validate_dimension(dimension: int) -> None:
        """Raises an exception if `dimension` is not positive.

        Raises:
            ValueError: `dimension` is not positive.
        """
        if dimension < 1:
            raise ValueError(
                f'Wrong qid dimension. Expected a positive integer but got {dimension}.'
            )

    def with_dimension(self, dimension: int) -> 'Qid':
        """Returns a new qid with a different dimension.

        Child classes can override.  Wraps the qubit object by default.

        Args:
            dimension: The new dimension or number of levels.
        """
        if dimension == self.dimension:
            return self
        return _QubitAsQid(self, dimension=dimension)

    def _cmp_tuple(self):
        return (type(self).__name__, repr(type(self)), self._comparison_key(), self.dimension)

    @cached_method
    def __hash__(self) -> int:
        return hash((Qid, self._comparison_key()))

    def __getstate__(self) -> Dict[str, Any]:
        # clear cached hash value when pickling, see #6674
        state = self.__dict__
        hash_attr = _method_cache_name(self.__hash__)
        if hash_attr in state:
            state = state.copy()
            del state[hash_attr]
        return state

    def __eq__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() == other._cmp_tuple()

    def __ne__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() != other._cmp_tuple()

    def __lt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __gt__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __le__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __ge__(self, other):
        if not isinstance(other, Qid):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        """Circuit symbol for qids defaults to the string representation."""
        return protocols.CircuitDiagramInfo(wire_symbols=(str(self),))


@functools.total_ordering
class _QubitAsQid(Qid):
    def __init__(self, qubit: Qid, dimension: int):
        self._qubit = qubit
        self._dimension = dimension
        self.validate_dimension(dimension)

    @property
    def qubit(self) -> Qid:
        return self._qubit

    @property
    def dimension(self) -> int:
        return self._dimension

    def with_dimension(self, dimension: int) -> Qid:
        """Returns a copy with a different dimension or number of levels."""
        return self.qubit.with_dimension(dimension)

    def _comparison_key(self) -> Any:
        # Don't include self._qubit.dimension
        return self._qubit._cmp_tuple()[:-1]

    def __repr__(self) -> str:
        return f'{self.qubit!r}.with_dimension({self.dimension})'

    def __str__(self) -> str:
        return f'{self.qubit!s} (d={self.dimension})'

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['qubit', 'dimension'])


class Gate(metaclass=value.ABCMetaImplementAnyOneOf):
    """An operation type that can be applied to a collection of qubits.

    Gates can be applied to qubits by calling their on() method with
    the qubits to be applied to supplied, or, alternatively, by simply
    calling the gate on the qubits.  In other words calling MyGate.on(q1, q2)
    to create an Operation on q1 and q2 is equivalent to MyGate(q1,q2).

    Gates operate on a certain number of qubits. All implementations of gate
    must implement the `num_qubits` method declaring how many qubits they
    act on.

    Linear combinations of gates can be created by adding gates together and
    multiplying them by scalars.
    """

    def validate_args(self, qubits: Sequence['cirq.Qid']) -> None:
        """Checks if this gate can be applied to the given qubits.

        By default checks that:
        - inputs are of type `Qid`
        - len(qubits) == num_qubits()
        - qubit_i.dimension == qid_shape[i] for all qubits

        Child classes can override.  The child implementation should call
        `super().validate_args(qubits)` then do custom checks.

        Args:
            qubits: The sequence of qubits to potentially apply the gate to.

        Raises:
            ValueError: The gate can't be applied to the qubits.
        """
        if __cirq_debug__.get():
            _validate_qid_shape(self, qubits)

    def on(self, *qubits: Qid) -> 'Operation':
        """Returns an application of this gate to the given qubits.

        Args:
            *qubits: The collection of qubits to potentially apply the gate to.

        Returns: a `cirq.Operation` which is this gate applied to the given
            qubits.
        """
        return ops.gate_operation.GateOperation(self, qubits)

    def on_each(self, *targets: Union[Qid, Iterable[Any]]) -> List['cirq.Operation']:
        """Returns a list of operations applying the gate to all targets.

        Args:
            *targets: The qubits to apply this gate to. For single-qubit gates
                this can be provided as varargs or a combination of nested
                iterables. For multi-qubit gates this must be provided as an
                `Iterable[Sequence[Qid]]`, where each sequence has `num_qubits`
                qubits.

        Returns:
            Operations applying this gate to the target qubits.

        Raises:
            ValueError: If targets are not instances of Qid or Iterable[Qid].
                If the gate qubit number is incompatible.
            TypeError: If a single target is supplied and it is not iterable.
        """
        operations: List['cirq.Operation'] = []
        if self._num_qubits_() > 1:
            iterator: Iterable = targets
            if len(targets) == 1:
                if not isinstance(targets[0], Iterable):
                    raise TypeError(f'{targets[0]} object is not iterable.')
                t0 = list(targets[0])
                iterator = [t0] if t0 and isinstance(t0[0], Qid) else t0
            if __cirq_debug__.get():
                for target in iterator:
                    if not isinstance(target, Sequence):
                        raise ValueError(
                            f'Inputs to multi-qubit gates must be Sequence[Qid].'
                            f' Type: {type(target)}'
                        )
                    if not all(isinstance(x, Qid) for x in target):
                        raise ValueError(f'All values in sequence should be Qids, but got {target}')
                    if len(target) != self._num_qubits_():
                        raise ValueError(f'Expected {self._num_qubits_()} qubits, got {target}')
                    operations.append(self.on(*target))
            else:
                operations = [self.on(*target) for target in iterator]
            return operations

        if not __cirq_debug__.get():
            return [
                op
                for q in targets
                for op in (
                    self.on_each(*q)
                    if isinstance(q, Iterable) and not isinstance(q, str)
                    else [self.on(cast('cirq.Qid', q))]
                )
            ]

        for target in targets:
            if isinstance(target, Qid):
                operations.append(self.on(target))
            elif isinstance(target, Iterable) and not isinstance(target, str):
                operations.extend(self.on_each(*target))
            else:
                raise ValueError(
                    f'Gate was called with type different than Qid. Type: {type(target)}'
                )
        return operations

    def wrap_in_linear_combination(
        self, coefficient: Union[complex, float, int] = 1
    ) -> 'cirq.LinearCombinationOfGates':
        """Returns a LinearCombinationOfGates with this gate.

        Args:
            coefficient: number coefficient to use in the resulting
                `cirq.LinearCombinationOfGates` object.

        Returns:
            `cirq.LinearCombinationOfGates` containing self with a
                coefficient of `coefficient`.
        """
        return ops.linear_combinations.LinearCombinationOfGates({self: coefficient})

    def __add__(
        self, other: Union['Gate', 'cirq.LinearCombinationOfGates']
    ) -> 'cirq.LinearCombinationOfGates':
        if isinstance(other, Gate):
            return self.wrap_in_linear_combination() + other.wrap_in_linear_combination()
        return self.wrap_in_linear_combination() + other

    def __sub__(
        self, other: Union['Gate', 'cirq.LinearCombinationOfGates']
    ) -> 'cirq.LinearCombinationOfGates':
        if isinstance(other, Gate):
            return self.wrap_in_linear_combination() - other.wrap_in_linear_combination()
        return self.wrap_in_linear_combination() - other

    def __neg__(self) -> 'cirq.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=-1)

    def __mul__(self, other: Union[complex, float, int]) -> 'cirq.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=other)

    def __rmul__(self, other: Union[complex, float, int]) -> 'cirq.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=other)

    def __truediv__(self, other: Union[complex, float, int]) -> 'cirq.LinearCombinationOfGates':
        return self.wrap_in_linear_combination(coefficient=1 / other)

    def __pow__(self, power):
        if power == 1:
            return self

        if power == -1:
            decomposed = protocols.decompose_once_with_qubits(
                self, qubits=line_qubit.LineQid.for_gate(self), default=None
            )
            if decomposed is None:
                return NotImplemented

            inverse_decomposed = protocols.inverse(decomposed, None)
            if inverse_decomposed is None:
                return NotImplemented

            return _InverseCompositeGate(self)

        return NotImplemented

    def __call__(self, *qubits: Qid, **kwargs):
        return self.on(*qubits)

    def with_probability(self, probability: 'cirq.TParamVal') -> 'cirq.Gate':
        """Creates a probabilistic channel with this gate.

        Args:
            probability: floating point value between 0 and 1, giving the
                probability this gate is applied.

        Returns:
            `cirq.RandomGateChannel` that applies `self` with probability
                `probability` and the identity with probability `1-p`.
        """
        if probability == 1:
            return self
        return ops.random_gate_channel.RandomGateChannel(sub_gate=self, probability=probability)

    def controlled(
        self,
        num_controls: Optional[int] = None,
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
        control_qid_shape: Optional[Tuple[int, ...]] = None,
    ) -> 'Gate':
        """Returns a controlled version of this gate. If no arguments are
        specified, defaults to a single qubit control.

        Args:
            num_controls: Total number of control qubits.
            control_values: Which control computational basis state to apply the
                sub gate.  A sequence of length `num_controls` where each
                entry is an integer (or set of integers) corresponding to the
                computational basis state (or set of possible values) where that
                control is enabled.  When all controls are enabled, the sub gate is
                applied.  If unspecified, control values default to 1.
            control_qid_shape: The qid shape of the controls.  A tuple of the
                expected dimension of each control qid.  Defaults to
                `(2,) * num_controls`.  Specify this argument when using qudits.

        Returns:
            A `cirq.Gate` representing `self` controlled by the given control values
                and qubits. This is a `cirq.ControlledGate` in the base
                implementation, but subclasses may return a different gate type.
        """

        if num_controls == 0:
            return self
        return ops.controlled_gate.ControlledGate(
            self,
            num_controls=num_controls,
            control_values=control_values,
            control_qid_shape=control_qid_shape,
        )

    # num_qubits, _num_qubits_, and _qid_shape_ are implemented with alternative
    # to keep backwards compatibility with versions of cirq where num_qubits
    # is an abstract method.
    def _backwards_compatibility_num_qubits(self) -> int:
        return protocols.num_qubits(self)

    def _has_stabilizer_effect_(self) -> Optional[bool]:
        return NotImplemented

    @value.alternative(requires='_num_qubits_', implementation=_backwards_compatibility_num_qubits)
    def num_qubits(self) -> int:
        """The number of qubits this gate acts on."""
        raise NotImplementedError

    def _num_qubits_from_shape(self) -> int:
        shape = self._qid_shape_()
        if shape is NotImplemented:
            return NotImplemented
        return len(shape)

    def _num_qubits_proto_from_num_qubits(self) -> int:
        return self.num_qubits()

    @value.alternative(requires='num_qubits', implementation=_num_qubits_proto_from_num_qubits)
    @value.alternative(requires='_qid_shape_', implementation=_num_qubits_from_shape)
    def _num_qubits_(self) -> int:
        """The number of qubits this gate acts on."""
        raise NotImplementedError

    def _default_shape_from_num_qubits(self) -> Tuple[int, ...]:
        num_qubits = self._num_qubits_()
        if num_qubits is NotImplemented:
            return NotImplemented
        return (2,) * num_qubits

    @value.alternative(requires='_num_qubits_', implementation=_default_shape_from_num_qubits)
    def _qid_shape_(self) -> Tuple[int, ...]:
        """Returns a Tuple containing the number of quantum levels of each qid
        the gate acts on.  E.g. (2, 2, 2) for the three-qubit CCZ gate and
        (3, 3) for a 2-qutrit ternary gate.
        """
        raise NotImplementedError

    def _commutes_on_qids_(
        self, qids: 'Sequence[cirq.Qid]', other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        return NotImplemented

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[None, NotImplementedType, bool]:
        if not isinstance(other, Gate):
            return NotImplemented
        if protocols.qid_shape(self) != protocols.qid_shape(other):
            return None

        qs = line_qubit.LineQid.for_qid_shape(protocols.qid_shape(self))
        return protocols.commutes(self(*qs), other(*qs))

    def _mul_with_qubits(self, qubits: Tuple['cirq.Qid', ...], other):
        """cirq.GateOperation.__mul__ delegates to this method."""
        return NotImplemented

    def _rmul_with_qubits(self, qubits: Tuple['cirq.Qid', ...], other):
        """cirq.GateOperation.__rmul__ delegates to this method."""
        return NotImplemented

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, attribute_names=[])


class Operation(metaclass=abc.ABCMeta):
    """An effect applied to a collection of qubits.

    The most common kind of Operation is a GateOperation, which separates its
    effect into a qubit-independent Gate and the qubits it should be applied to.
    """

    @property
    def gate(self) -> Optional['cirq.Gate']:
        return None

    @property
    @abc.abstractmethod
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        raise NotImplementedError()

    def _num_qubits_(self) -> int:
        """The number of qubits this operation acts on.

        By definition, returns the length of `qubits`.
        """
        return len(self.qubits)

    @cached_method
    def _qid_shape_(self) -> Tuple[int, ...]:
        return protocols.qid_shape(self.qubits)

    @abc.abstractmethod
    def with_qubits(self, *new_qubits: 'cirq.Qid') -> Self:
        """Returns the same operation, but applied to different qubits.

        Args:
            *new_qubits: The new qubits to apply the operation to. The order must
                exactly match the order of qubits returned from the operation's
                `qubits` property.
        """

    @property
    def tags(self) -> Tuple[Hashable, ...]:
        """Returns a tuple of the operation's tags."""
        return ()

    @property
    def untagged(self) -> 'cirq.Operation':
        """Returns the underlying operation without any tags."""
        return self

    def with_tags(self, *new_tags: Hashable) -> 'cirq.Operation':
        """Creates a new TaggedOperation, with this op and the specified tags.

        This method can be used to attach meta-data to specific operations
        without affecting their functionality.  The intended usage is to
        attach classes intended for this purpose or strings to mark operations
        for specific usage that will be recognized by consumers.  Specific
        examples include ignoring this operation in optimization passes,
        hardware-specific functionality, or circuit diagram customizability.

        Tags can be a list of any type of object that is useful to identify
        this operation as long as the type is hashable.  If you wish the
        resulting operation to be eventually serialized into JSON, you should
        also restrict the operation to be JSON serializable.

        Args:
            *new_tags: The tags to wrap this operation in.
        """
        if not new_tags:
            return self
        return TaggedOperation(self, *new_tags)

    def transform_qubits(
        self, qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]
    ) -> Self:
        """Returns the same operation, but with different qubits.

        Args:
            qubit_map: A function or a dict mapping each current qubit into a desired
                new qubit.

        Returns:
            The receiving operation but with qubits transformed by the given
                function.
        Raises:
            TypeError: qubit_map was not a function or dict mapping qubits to
                qubits.
        """
        if callable(qubit_map):
            transform = qubit_map
        elif isinstance(qubit_map, dict):
            transform = lambda q: qubit_map.get(q, q)
        else:
            raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')
        return self.with_qubits(*(transform(q) for q in self.qubits))

    def controlled_by(
        self,
        *control_qubits: 'cirq.Qid',
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
    ) -> 'cirq.Operation':
        """Returns a controlled version of this operation. If no control_qubits
           are specified, returns self.

        Args:
            *control_qubits: Qubits to control the operation by. Required.
            control_values: For which control qubit values to apply the
                operation.  A sequence of the same length as `control_qubits`
                where each entry is an integer (or set of integers)
                corresponding to the qubit value (or set of possible values)
                where that control is enabled.  When all controls are enabled,
                the operation is applied.  If unspecified, control values
                default to 1.
        """
        if len(control_qubits) == 0:
            return self
        return ops.controlled_operation.ControlledOperation(control_qubits, self, control_values)

    def with_probability(self, probability: 'cirq.TParamVal') -> 'cirq.Operation':
        """Creates a probabilistic channel with this operation.

        Args:
            probability: floating point value between 0 and 1, giving the
                probability this gate is applied.

        Returns:
            `cirq.RandomGateChannel` that applies `self` with probability
                `probability` and the identity with probability `1-p`.

        Raises:
            NotImplementedError: if called on an operation that lacks a gate.
        """
        gate = self.gate
        if gate is None:
            raise NotImplementedError("with_probability on gateless operation.")
        if probability == 1:
            return self
        return ops.random_gate_channel.RandomGateChannel(sub_gate=gate, probability=probability).on(
            *self.qubits
        )

    def validate_args(self, qubits: Sequence['cirq.Qid']):
        """Raises an exception if the `qubits` don't match this operation's qid
        shape.

        Call this method from a subclass's `with_qubits` method.

        Args:
            qubits: The new qids for the operation.

        Raises:
            ValueError: The operation had qids that don't match it's qid shape.
        """
        if __cirq_debug__.get():
            _validate_qid_shape(self, qubits)

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        """Determine if this Operation commutes with the object"""
        if not isinstance(other, Operation):
            return NotImplemented

        self_keys = protocols.measurement_key_objs(self)
        other_keys = protocols.measurement_key_objs(other)
        if (
            not self_keys.isdisjoint(other_keys)
            or not protocols.control_keys(self).isdisjoint(other_keys)
            or not protocols.control_keys(other).isdisjoint(self_keys)
        ):
            return False

        if hasattr(other, 'qubits') and set(self.qubits).isdisjoint(other.qubits):
            return True

        from cirq import circuits

        # Remove the classical controls to validate the quantum commutativity. This can be done
        # because during execution, the two operations will either both be run, in which case they
        # behave like the suboperations, so if the suboperations commute then these commute. Or
        # one of them is cold in which case it behaves like the identity, which always commutes.
        self_raw = self.without_classical_controls()
        other_raw = other.without_classical_controls()
        circuit12 = circuits.Circuit(self_raw, other_raw)
        circuit21 = circuits.Circuit(other_raw, self_raw)

        # Don't create gigantic matrices.
        shape = protocols.qid_shape_protocol.qid_shape(circuit12)
        if np.prod(shape, dtype=np.int64) > 2**10:
            return NotImplemented  # pragma: no cover

        m12 = protocols.unitary_protocol.unitary(circuit12, default=None)
        m21 = protocols.unitary_protocol.unitary(circuit21, default=None)
        if m12 is None or m21 is None:
            return NotImplemented

        return np.allclose(m12, m21, atol=atol)

    @property
    def classical_controls(self) -> FrozenSet['cirq.Condition']:
        """The classical controls gating this operation."""
        return frozenset()

    def with_classical_controls(
        self, *conditions: Union[str, 'cirq.MeasurementKey', 'cirq.Condition', sympy.Expr]
    ) -> 'cirq.Operation':
        """Returns a classically controlled version of this operation.

        An operation that is classically controlled is executed iff all
        conditions evaluate to True. Conditions can be either a measurement key
        or a user-specified `cirq.Condition`. A measurement key evaluates to
        True iff any qubit in the corresponding measurement operation evaluated
        to a non-zero value; `cirq.Condition` supports more complex,
        user-defined conditions.

        If no conditions are specified, returns self.

        The classical control will remove any tags on the existing operation,
        since tags are fragile, and we always opt to get rid of the tags when
        the underlying operation is changed.

        Args:
            *conditions: A list of measurement keys, strings that can be parsed
                into measurement keys, or sympy expressions where the free
                symbols are measurement key strings.

        Returns:
            A `ClassicallyControlledOperation` wrapping the operation. If no conditions
           are specified, returns self.
        """
        from cirq.ops.classically_controlled_operation import ClassicallyControlledOperation

        if not conditions:
            return self
        return ClassicallyControlledOperation(self, conditions)

    def without_classical_controls(self) -> 'cirq.Operation':
        """Removes all classical controls from the operation.

        This function removes all classical controls gating the operation. It
        acts recursively, so that all classical control wrappers are always
        removed from the current operation.

        If there are no classical controls on the operation, it will return
        `self`.

        Since tags are fragile, this will also remove any tags from the operation,
        when called on `TaggedOperation` (unless there are no classical controls on it).
        If a `TaggedOperation` is under all the classical control layers,
        that `TaggedOperation` will be returned from this function.

        Returns:
            The operation with all classical controls removed.
        """
        return self


@value.value_equality
class TaggedOperation(Operation):
    """Operation annotated with a set of tags.

    These Tags can be used for special processing.  TaggedOperations
    can be initialized with using `Operation.with_tags(tag)`
    or by using `TaggedOperation(op, tag)`.

    Tags added can be of any type, but they should be Hashable in order
    to allow equality checking.  If you wish to serialize operations into
    JSON, you should restrict yourself to only use objects that have a JSON
    serialization.

    See `Operation.with_tags()` for more information on intended usage.
    """

    def __init__(self, sub_operation: 'cirq.Operation', *tags: Hashable):
        self._sub_operation = sub_operation
        self._tags = tuple(tags)

    @property
    def sub_operation(self) -> 'cirq.Operation':
        return self._sub_operation

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        return self.sub_operation.qubits

    @property
    def gate(self) -> Optional['cirq.Gate']:
        return self.sub_operation.gate

    def with_qubits(self, *new_qubits: 'cirq.Qid'):
        return TaggedOperation(self.sub_operation.with_qubits(*new_qubits), *self._tags)

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]):
        sub_op = protocols.with_measurement_key_mapping(self.sub_operation, key_map)
        if sub_op is NotImplemented:
            return NotImplemented
        return TaggedOperation(sub_op, *self.tags)

    def controlled_by(
        self,
        *control_qubits: 'cirq.Qid',
        control_values: Optional[
            Union[cv.AbstractControlValues, Sequence[Union[int, Collection[int]]]]
        ] = None,
    ) -> 'cirq.Operation':
        if len(control_qubits) == 0:
            return self
        return self.sub_operation.controlled_by(*control_qubits, control_values=control_values)

    @property
    def tags(self) -> Tuple[Hashable, ...]:
        """Returns a tuple of the operation's tags."""
        return self._tags

    @property
    def untagged(self) -> 'cirq.Operation':
        """Returns the underlying operation without any tags."""
        return self.sub_operation

    def with_tags(self, *new_tags: Hashable) -> 'cirq.TaggedOperation':
        """Creates a new TaggedOperation with combined tags.

        Overloads Operation.with_tags to create a new TaggedOperation
        that has the tags of this operation combined with the new_tags
        specified as the parameter.
        """
        if not new_tags:
            return self
        return TaggedOperation(self.sub_operation, *self._tags, *new_tags)

    def __str__(self) -> str:
        tag_repr = ','.join(repr(t) for t in self._tags)
        return f'cirq.TaggedOperation({repr(self.sub_operation)}, {tag_repr})'

    def __repr__(self) -> str:
        return str(self)

    def _value_equality_values_(self) -> Any:
        return (self.sub_operation, self._tags)

    @classmethod
    def _from_json_dict_(cls, sub_operation, tags, **kwargs):
        return cls(sub_operation, *tags)

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['sub_operation', 'tags'])

    def _decompose_(self) -> 'cirq.OP_TREE':
        return self._decompose_with_context_()

    def _decompose_with_context_(
        self, context: Optional['cirq.DecompositionContext'] = None
    ) -> 'cirq.OP_TREE':
        return protocols.decompose_once(
            self.sub_operation, default=None, flatten=False, context=context
        )

    def _pauli_expansion_(self) -> value.LinearDict[str]:
        return protocols.pauli_expansion(self.sub_operation)

    def _apply_unitary_(
        self, args: 'protocols.ApplyUnitaryArgs'
    ) -> Union[np.ndarray, None, NotImplementedType]:
        return protocols.apply_unitary(self.sub_operation, args, default=None)

    @cached_method
    def _has_unitary_(self) -> bool:
        return protocols.has_unitary(self.sub_operation)

    def _unitary_(self) -> Union[np.ndarray, NotImplementedType]:
        return protocols.unitary(self.sub_operation, NotImplemented)

    def _commutes_(
        self, other: Any, *, atol: float = 1e-8
    ) -> Union[bool, NotImplementedType, None]:
        return protocols.commutes(self.sub_operation, other, atol=atol)

    @cached_method
    def _has_mixture_(self) -> bool:
        return protocols.has_mixture(self.sub_operation)

    def _mixture_(self) -> Sequence[Tuple[float, Any]]:
        return protocols.mixture(self.sub_operation, NotImplemented)

    @cached_method
    def _has_kraus_(self) -> bool:
        return protocols.has_kraus(self.sub_operation)

    def _kraus_(self) -> Union[Tuple[np.ndarray], NotImplementedType]:
        return protocols.kraus(self.sub_operation, NotImplemented)

    @cached_method
    def _measurement_key_names_(self) -> FrozenSet[str]:
        return protocols.measurement_key_names(self.sub_operation)

    @cached_method
    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        return protocols.measurement_key_objs(self.sub_operation)

    @cached_method
    def _is_measurement_(self) -> bool:
        sub = getattr(self.sub_operation, "_is_measurement_", None)
        if sub is not None:
            return sub()
        return NotImplemented

    @cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self.sub_operation) or any(
            protocols.is_parameterized(tag) for tag in self.tags
        )

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> bool:
        sub = getattr(self.sub_operation, "_act_on_", None)
        if sub is not None:
            return sub(sim_state)
        return NotImplemented

    @cached_method
    def _parameter_names_(self) -> AbstractSet[str]:
        tag_params = {name for tag in self.tags for name in protocols.parameter_names(tag)}
        return protocols.parameter_names(self.sub_operation) | tag_params

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'TaggedOperation':
        resolved_op = protocols.resolve_parameters(self.sub_operation, resolver, recursive)
        resolved_tags = (
            protocols.resolve_parameters(tag, resolver, recursive) for tag in self._tags
        )
        return TaggedOperation(resolved_op, *resolved_tags)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        sub_op_info = protocols.circuit_diagram_info(self.sub_operation, args, NotImplemented)
        # Add tag to wire symbol if it exists.
        if sub_op_info is not NotImplemented and args.include_tags and sub_op_info.wire_symbols:
            sub_op_info.wire_symbols = (
                sub_op_info.wire_symbols[0] + f"[{', '.join(map(str, self._tags))}]",
            ) + sub_op_info.wire_symbols[1:]
        return sub_op_info

    @cached_method
    def _trace_distance_bound_(self) -> float:
        return protocols.trace_distance_bound(self.sub_operation)

    def _phase_by_(self, phase_turns: float, qubit_index: int) -> 'cirq.Operation':
        return protocols.phase_by(self.sub_operation, phase_turns, qubit_index)

    def __pow__(self, exponent: Any) -> 'cirq.Operation':
        return self.sub_operation**exponent

    def __mul__(self, other: Any) -> Any:
        return self.sub_operation * other

    def __rmul__(self, other: Any) -> Any:
        return other * self.sub_operation

    def _qasm_(self, args: 'protocols.QasmArgs') -> Optional[str]:
        return protocols.qasm(self.sub_operation, args=args, default=None)

    def _equal_up_to_global_phase_(
        self, other: Any, atol: Union[int, float] = 1e-8
    ) -> Union[NotImplementedType, bool]:
        return protocols.equal_up_to_global_phase(self.sub_operation, other, atol=atol)

    @property
    def classical_controls(self) -> FrozenSet['cirq.Condition']:
        return self.sub_operation.classical_controls

    def without_classical_controls(self) -> 'cirq.Operation':
        new_sub_operation = self.sub_operation.without_classical_controls()
        return self if new_sub_operation is self.sub_operation else new_sub_operation

    def with_classical_controls(
        self, *conditions: Union[str, 'cirq.MeasurementKey', 'cirq.Condition', sympy.Expr]
    ) -> 'cirq.Operation':
        if not conditions:
            return self
        return self.sub_operation.with_classical_controls(*conditions)

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        return protocols.control_keys(self.sub_operation)


@value.value_equality
class _InverseCompositeGate(Gate):
    """The inverse of a composite gate."""

    def __init__(self, original: Gate) -> None:
        self._original = original

    def _qid_shape_(self):
        return protocols.qid_shape(self._original)

    def __pow__(self, power):
        if power == 1:
            return self
        if power == -1:
            return self._original
        return NotImplemented

    def _decompose_(self, qubits):
        return self._decompose_with_context_(qubits)

    def _decompose_with_context_(
        self, qubits: Sequence['cirq.Qid'], context: Optional['cirq.DecompositionContext'] = None
    ) -> 'cirq.OP_TREE':
        return protocols.inverse(
            protocols.decompose_once_with_qubits(self._original, qubits, context=context)
        )

    def _has_unitary_(self):
        from cirq import protocols, devices

        qubits = devices.LineQid.for_gate(self)
        return all(
            protocols.has_unitary(op)
            for op in protocols.decompose_once_with_qubits(self._original, qubits)
        )

    @cached_method
    def _is_parameterized_(self) -> bool:
        return protocols.is_parameterized(self._original)

    @cached_method
    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self._original)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> '_InverseCompositeGate':
        return _InverseCompositeGate(
            protocols.resolve_parameters(self._original, resolver, recursive)
        )

    def _value_equality_values_(self):
        return self._original

    def _circuit_diagram_info_(self, args: 'cirq.CircuitDiagramInfoArgs'):
        sub_info = protocols.circuit_diagram_info(self._original, args, default=NotImplemented)
        if sub_info is NotImplemented:
            return NotImplemented
        sub_info.exponent *= -1
        return sub_info

    def __repr__(self) -> str:
        return f'({self._original!r}**-1)'

    def __str__(self) -> str:
        return f'{self._original!s}†'


def _validate_qid_shape(val: Any, qubits: Sequence['cirq.Qid']) -> None:
    """Helper function to validate qubits for gates and operations.

    Raises:
        ValueError: The operation had qids that don't match it's qid shape.
    """
    qid_shape = protocols.qid_shape(val)
    if len(qubits) != len(qid_shape):
        raise ValueError(
            f'Wrong number of qubits for <{val!r}>. '
            f'Expected {len(qid_shape)} qubits but got <{qubits!r}>.'
        )
    if any(qid.dimension != dimension for qid, dimension in zip(qubits, qid_shape)):
        raise ValueError(
            f'Wrong shape of qids for <{val!r}>. '
            f'Expected {qid_shape} but got {tuple(qid.dimension for qid in qubits)} <{qubits!r}>.'
        )
    if len(set(qubits)) != len(qubits):
        raise ValueError(
            f'Duplicate qids for <{val!r}>. Expected unique qids but got <{qubits!r}>.'
        )
