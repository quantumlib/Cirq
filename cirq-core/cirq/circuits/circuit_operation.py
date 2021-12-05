# Copyright 2020 The Cirq Developers
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
"""A structure for encapsulating entire circuits in an operation.

A CircuitOperation is an Operation object that wraps a FrozenCircuit. When
applied as part of a larger circuit, a CircuitOperation will execute all
component operations in order, including any nested CircuitOperations.
"""
from typing import (
    TYPE_CHECKING,
    AbstractSet,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Iterator,
    Sequence,
)

import dataclasses
import numpy as np

from cirq import circuits, ops, protocols, value, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


INT_TYPE = Union[int, np.integer]
REPETITION_ID_SEPARATOR = '-'


def default_repetition_ids(repetitions: int) -> Optional[List[str]]:
    if abs(repetitions) > 1:
        return [str(i) for i in range(abs(repetitions))]
    return None


def _full_join_string_lists(list1: Optional[List[str]], list2: Optional[List[str]]):
    if list1 is None and list2 is None:
        return None  # coverage: ignore
    if list1 is None:
        return list2  # coverage: ignore
    if list2 is None:
        return list1
    return [
        f'{REPETITION_ID_SEPARATOR.join([first, second])}' for first in list1 for second in list2
    ]


@dataclasses.dataclass(frozen=True)
class CircuitOperation(ops.Operation):
    """An operation that encapsulates a circuit.

    This class captures modifications to the contained circuit, such as tags
    and loops, to support more condensed serialization. Similar to
    GateOperation, this type is immutable.

    Args:
        circuit: The FrozenCircuit wrapped by this operation.
        repetitions: How many times the circuit should be repeated.
        qubit_map: Remappings for qubits in the circuit.
        measurement_key_map: Remappings for measurement keys in the circuit.
            The keys and values should be unindexed (i.e. without repetition_ids).
            The values cannot contain the `MEASUREMENT_KEY_SEPARATOR`.
        param_resolver: Resolved values for parameters in the circuit.
        parent_path: A tuple of identifiers for any parent CircuitOperations containing this one.
        repetition_ids: List of identifiers for each repetition of the
            CircuitOperation. If populated, the length should be equal to the
            repetitions. If not populated and abs(`repetitions`) > 1, it is
            initialized to strings for numbers in `range(repetitions)`.
    """

    _hash: Optional[int] = dataclasses.field(default=None, init=False)
    _cached_measurement_key_objs: Optional[AbstractSet[value.MeasurementKey]] = dataclasses.field(
        default=None, init=False
    )

    circuit: 'cirq.FrozenCircuit'
    repetitions: int = 1
    qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = dataclasses.field(default_factory=dict)
    measurement_key_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    param_resolver: study.ParamResolver = study.ParamResolver()
    repetition_ids: Optional[List[str]] = dataclasses.field(default=None)
    parent_path: Tuple[str, ...] = dataclasses.field(default_factory=tuple)

    def __post_init__(self):
        if not isinstance(self.circuit, circuits.FrozenCircuit):
            raise TypeError(f'Expected circuit of type FrozenCircuit, got: {type(self.circuit)!r}')

        # Ensure that the circuit is invertible if the repetitions are negative.
        if self.repetitions < 0:
            try:
                protocols.inverse(self.circuit.unfreeze())
            except TypeError:
                raise ValueError('repetitions are negative but the circuit is not invertible')

        # Initialize repetition_ids to default, if unspecified. Else, validate their length.
        loop_size = abs(self.repetitions)
        if not self.repetition_ids:
            object.__setattr__(self, 'repetition_ids', self._default_repetition_ids())
        elif len(self.repetition_ids) != loop_size:
            raise ValueError(
                f'Expected repetition_ids to be a list of length {loop_size}, '
                f'got: {self.repetition_ids}'
            )

        # Disallow mapping to keys containing the `MEASUREMENT_KEY_SEPARATOR`
        for mapped_key in self.measurement_key_map.values():
            if value.MEASUREMENT_KEY_SEPARATOR in mapped_key:
                raise ValueError(
                    f'Mapping to invalid key: {mapped_key}. "{value.MEASUREMENT_KEY_SEPARATOR}" '
                    'is not allowed for measurement keys in a CircuitOperation'
                )

        # Disallow qid mapping dimension conflicts.
        for q, q_new in self.qubit_map.items():
            if q_new.dimension != q.dimension:
                raise ValueError(f'Qid dimension conflict.\nFrom qid: {q}\nTo qid: {q_new}')

        # Ensure that param_resolver is converted to an actual ParamResolver.
        object.__setattr__(self, 'param_resolver', study.ParamResolver(self.param_resolver))

    def base_operation(self) -> 'CircuitOperation':
        """Returns a copy of this operation with only the wrapped circuit.

        Key and qubit mappings, parameter values, and repetitions are not copied.
        """
        return CircuitOperation(self.circuit)

    def replace(self, **changes) -> 'CircuitOperation':
        """Returns a copy of this operation with the specified changes."""
        return dataclasses.replace(self, **changes)

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.circuit == other.circuit
            and self.qubit_map == other.qubit_map
            and self.measurement_key_map == other.measurement_key_map
            and self.param_resolver == other.param_resolver
            and self.repetitions == other.repetitions
            and self.repetition_ids == other.repetition_ids
            and self.parent_path == other.parent_path
        )

    # Methods for getting post-mapping properties of the contained circuit.

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Returns the qubits operated on by this object."""
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(self.circuit.all_qubits())
        return tuple(self.qubit_map.get(q, q) for q in ordered_qubits)

    def _default_repetition_ids(self) -> Optional[List[str]]:
        return default_repetition_ids(self.repetitions)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple(q.dimension for q in self.qubits)

    def _is_measurement_(self) -> bool:
        return self.circuit._is_measurement_()

    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        if self._cached_measurement_key_objs is None:
            circuit_keys = protocols.measurement_key_objs(self.circuit)
            if self.repetition_ids is not None:
                circuit_keys = {
                    key.with_key_path_prefix(repetition_id)
                    for repetition_id in self.repetition_ids
                    for key in circuit_keys
                }
            circuit_keys = {
                protocols.with_key_path_prefix(key, self.parent_path) for key in circuit_keys
            }
            object.__setattr__(
                self,
                '_cached_measurement_key_objs',
                {
                    protocols.with_measurement_key_mapping(key, self.measurement_key_map)
                    for key in circuit_keys
                },
            )
        return self._cached_measurement_key_objs  # type: ignore

    def _measurement_key_names_(self) -> AbstractSet[str]:
        return {str(key) for key in self._measurement_key_objs_()}

    def _parameter_names_(self) -> AbstractSet[str]:
        return {
            name
            for symbol in protocols.parameter_symbols(self.circuit)
            for name in protocols.parameter_names(
                protocols.resolve_parameters(symbol, self.param_resolver, recursive=False)
            )
        }

    def mapped_circuit(self, deep: bool = False) -> 'cirq.Circuit':
        """Applies all maps to the contained circuit and returns the result.

        Args:
            deep: If true, this will also call mapped_circuit on any
                CircuitOperations this object contains.

        Returns:
            The contained circuit with all other member variables (repetitions,
            qubit mapping, parameterization, etc.) applied to it. This behaves
            like `cirq.decompose(self)`, but preserving moment structure.
        """
        circuit = self.circuit.unfreeze()
        circuit = circuit.transform_qubits(lambda q: self.qubit_map.get(q, q))
        if self.repetitions < 0:
            circuit = circuit ** -1
        has_measurements = protocols.is_measurement(circuit)
        if has_measurements:
            circuit = protocols.with_measurement_key_mapping(circuit, self.measurement_key_map)
        circuit = protocols.resolve_parameters(circuit, self.param_resolver, recursive=False)
        if deep:
            circuit = circuit.map_operations(
                lambda op: op.mapped_circuit(deep=True) if isinstance(op, CircuitOperation) else op
            )
        if self.repetition_ids:
            if not has_measurements:
                circuit = circuit * abs(self.repetitions)
            else:
                circuit = circuits.Circuit(
                    protocols.with_key_path_prefix(circuit, (rep,)) for rep in self.repetition_ids
                )
        if self.parent_path:
            circuit = protocols.with_key_path_prefix(circuit, self.parent_path)
        return circuit

    def mapped_op(self, deep: bool = False) -> 'cirq.CircuitOperation':
        """As `mapped_circuit`, but wraps the result in a CircuitOperation."""
        return CircuitOperation(circuit=self.mapped_circuit(deep=deep).freeze())

    def _decompose_(self) -> Iterator['cirq.Operation']:
        return self.mapped_circuit(deep=False).all_operations()

    def _act_on_(self, args: 'cirq.ActOnArgs') -> bool:
        for op in self._decompose_():
            protocols.act_on(op, args)
        return True

    # Methods for string representation of the operation.

    def __repr__(self):
        args = f'\ncircuit={self.circuit!r},\n'
        if self.repetitions != 1:
            args += f'repetitions={self.repetitions},\n'
        if self.qubit_map:
            args += f'qubit_map={proper_repr(self.qubit_map)},\n'
        if self.measurement_key_map:
            args += f'measurement_key_map={proper_repr(self.measurement_key_map)},\n'
        if self.param_resolver:
            args += f'param_resolver={proper_repr(self.param_resolver)},\n'
        if self.parent_path:
            args += f'parent_path={proper_repr(self.parent_path)},\n'
        if self.repetition_ids != self._default_repetition_ids():
            # Default repetition_ids need not be specified.
            args += f'repetition_ids={proper_repr(self.repetition_ids)},\n'
        indented_args = args.replace('\n', '\n    ')
        return f'cirq.CircuitOperation({indented_args[:-4]})'

    def __str__(self):
        # TODO: support out-of-line subcircuit definition in string format.
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(line) for line in msg_lines])
        circuit_msg = '\n'.join(
            '[ {line:<{width}} ]'.format(line=line, width=msg_width) for line in msg_lines
        )
        args = []

        def dict_str(d: Dict) -> str:
            pairs = [f'{k}: {v}' for k, v in sorted(d.items())]
            return '{' + ', '.join(pairs) + '}'

        if self.qubit_map:
            args.append(f'qubit_map={dict_str(self.qubit_map)}')
        if self.measurement_key_map:
            args.append(f'key_map={dict_str(self.measurement_key_map)}')
        if self.param_resolver:
            args.append(f'params={self.param_resolver.param_dict}')
        if self.parent_path:
            args.append(f'parent_path={self.parent_path}')
        if self.repetition_ids != self._default_repetition_ids():
            # Default repetition_ids need not be specified.
            args.append(f'repetition_ids={self.repetition_ids}')
        elif self.repetitions != 1:
            # Only add loops if we haven't added repetition_ids.
            args.append(f'loops={self.repetitions}')
        if not args:
            return circuit_msg
        return f'{circuit_msg}({", ".join(args)})'

    def __hash__(self):
        if self._hash is None:
            object.__setattr__(
                self,
                '_hash',
                hash(
                    (
                        self.circuit,
                        self.repetitions,
                        frozenset(self.qubit_map.items()),
                        frozenset(self.measurement_key_map.items()),
                        self.param_resolver,
                        self.parent_path,
                        tuple([] if self.repetition_ids is None else self.repetition_ids),
                    )
                ),
            )
        return self._hash

    def _json_dict_(self):
        return {
            'circuit': self.circuit,
            'repetitions': self.repetitions,
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'qubit_map': sorted(self.qubit_map.items()),
            'measurement_key_map': self.measurement_key_map,
            'param_resolver': self.param_resolver,
            'repetition_ids': self.repetition_ids,
            'parent_path': self.parent_path,
        }

    @classmethod
    def _from_json_dict_(
        cls,
        circuit,
        repetitions,
        qubit_map,
        measurement_key_map,
        param_resolver,
        repetition_ids,
        parent_path=(),
        **kwargs,
    ):
        return (
            cls(circuit)
            .with_qubit_mapping(dict(qubit_map))
            .with_measurement_key_mapping(measurement_key_map)
            .with_params(param_resolver)
            .with_key_path(tuple(parent_path))
            .repeat(repetitions, repetition_ids)
        )

    # Methods for constructing a similar object with one field modified.

    def repeat(
        self,
        repetitions: Optional[INT_TYPE] = None,
        repetition_ids: Optional[List[str]] = None,
    ) -> 'CircuitOperation':
        """Returns a copy of this operation repeated 'repetitions' times.
         Each repetition instance will be identified by a single repetition_id.

        Args:
            repetitions: Number of times this operation should repeat. This
                is multiplied with any pre-existing repetitions. If unset, it
                defaults to the length of `repetition_ids`.
            repetition_ids: List of IDs, one for each repetition. If unset,
                defaults to `default_repetition_ids(repetitions)`.

        Returns:
            A copy of this operation repeated `repetitions` times with the
            appropriate `repetition_ids`. The output `repetition_ids` are the
            cartesian product of input `repetition_ids` with the base
            operation's `repetition_ids`. If the base operation has unset
            `repetition_ids` (indicates {-1, 0, 1} `repetitions` with no custom
            IDs), the input `repetition_ids` are directly used.

        Raises:
            TypeError: `repetitions` is not an integer value.
            ValueError: Unexpected length of `repetition_ids`.
            ValueError: Both `repetitions` and `repetition_ids` are None.
        """
        if repetitions is None:
            if repetition_ids is None:
                raise ValueError('At least one of repetitions and repetition_ids must be set')
            repetitions = len(repetition_ids)

        if not isinstance(repetitions, (int, np.integer)):
            raise TypeError('Only integer repetitions are allowed.')

        repetitions = int(repetitions)

        if repetitions == 1 and repetition_ids is None:
            # As CircuitOperation is immutable, this can safely return the original.
            return self

        expected_repetition_id_length = abs(repetitions)
        # The eventual number of repetitions of the returned CircuitOperation.
        final_repetitions = self.repetitions * repetitions

        if repetition_ids is None:
            repetition_ids = default_repetition_ids(expected_repetition_id_length)
        elif len(repetition_ids) != expected_repetition_id_length:
            raise ValueError(
                f'Expected repetition_ids={repetition_ids} length to be '
                f'{expected_repetition_id_length}'
            )

        # If `self.repetition_ids` is None, this will just return `repetition_ids`.
        repetition_ids = _full_join_string_lists(repetition_ids, self.repetition_ids)

        return self.replace(repetitions=final_repetitions, repetition_ids=repetition_ids)

    def __pow__(self, power: int) -> 'CircuitOperation':
        return self.repeat(power)

    def _with_key_path_(self, path: Tuple[str, ...]):
        return dataclasses.replace(self, parent_path=path)

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return dataclasses.replace(self, parent_path=prefix + self.parent_path)

    def with_key_path(self, path: Tuple[str, ...]):
        return self._with_key_path_(path)

    def with_repetition_ids(self, repetition_ids: List[str]) -> 'CircuitOperation':
        return self.replace(repetition_ids=repetition_ids)

    def with_qubit_mapping(
        self,
        qubit_map: Union[Dict['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']],
    ) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated qubit mapping.

        Users should pass either 'qubit_map' or 'transform' to this method.

        Args:
            qubit_map: A mapping of old qubits to new qubits. This map will be
                composed with any existing qubit mapping.
            transform: A function mapping old qubits to new qubits. This
                function will be composed with any existing qubit mapping.

        Returns:
            A copy of this operation targeting qubits as indicated by qubit_map.

        Raises:
            TypeError: qubit_map was not a function or dict mapping qubits to
                qubits.
            ValueError: The new operation has a different number of qubits than
                this operation.
        """
        if callable(qubit_map):
            transform = qubit_map
        elif isinstance(qubit_map, dict):
            transform = lambda q: qubit_map.get(q, q)  # type: ignore
        else:
            raise TypeError('qubit_map must be a function or dict mapping qubits to qubits.')
        new_map = {}
        for q in self.circuit.all_qubits():
            q_new = transform(self.qubit_map.get(q, q))
            if q_new != q:
                if q_new.dimension != q.dimension:
                    raise ValueError(f'Qid dimension conflict.\nFrom qid: {q}\nTo qid: {q_new}')
                new_map[q] = q_new
        new_op = self.replace(qubit_map=new_map)
        if len(set(new_op.qubits)) != len(set(self.qubits)):
            raise ValueError(
                f'Collision in qubit map composition. Original map:\n{self.qubit_map}'
                f'\nMap after changes: {new_op.qubit_map}'
            )
        return new_op

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'CircuitOperation':
        """Returns a copy of this operation with an updated qubit mapping.

        Args:
            new_qubits: A list of qubits to target. Qubits in this list are
                matched to qubits in the circuit following default qubit order,
                ignoring any existing qubit map.

        Returns:
            A copy of this operation targeting `new_qubits`.

        Raises:
            ValueError: `new_qubits` has a different number of qubits than
                this operation.
        """
        expected = protocols.num_qubits(self.circuit)
        if len(new_qubits) != expected:
            raise ValueError(f'Expected {expected} qubits, got {len(new_qubits)}.')
        return self.with_qubit_mapping(dict(zip(self.qubits, new_qubits)))

    def with_measurement_key_mapping(self, key_map: Dict[str, str]) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated key mapping.

        Args:
            key_map: A mapping of old measurement keys to new measurement keys.
                This map will be composed with any existing key mapping.
                The keys and values of the map should be unindexed (i.e. without
                repetition_ids).

        Returns:
            A copy of this operation with measurement keys updated as specified
                by key_map.

        Raises:
            ValueError: The new operation has a different number of measurement
                keys than this operation.
        """
        new_map = {}
        for k_obj in self.circuit.all_measurement_key_objs():
            k = k_obj.name
            k_new = self.measurement_key_map.get(k, k)
            k_new = key_map.get(k_new, k_new)
            if k_new != k:
                new_map[k] = k_new
        new_op = self.replace(measurement_key_map=new_map)
        if len(new_op._measurement_key_objs_()) != len(self._measurement_key_objs_()):
            raise ValueError(
                f'Collision in measurement key map composition. Original map:\n'
                f'{self.measurement_key_map}\nApplied changes: {key_map}'
            )
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'CircuitOperation':
        return self.with_measurement_key_mapping(key_map)

    def with_params(self, param_values: study.ParamResolverOrSimilarType) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated ParamResolver.

        Note that any resulting parameter mappings with no corresponding
        parameter in the base circuit will be omitted.

        Args:
            param_values: A map or ParamResolver able to convert old param
                values to new param values. This map will be composed with any
                existing ParamResolver via single-step resolution.

        Returns:
            A copy of this operation with its ParamResolver updated as specified
                by param_values.
        """
        new_params = {}
        for k in protocols.parameter_symbols(self.circuit):
            v = self.param_resolver.value_of(k, recursive=False)
            v = protocols.resolve_parameters(v, param_values, recursive=False)
            if v != k:
                new_params[k] = v
        return self.replace(param_resolver=new_params)

    # TODO: handle recursive parameter resolution gracefully
    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'CircuitOperation':
        if recursive:
            raise ValueError(
                'Recursive resolution of CircuitOperation parameters is prohibited. '
                'Use "recursive=False" to prevent this error.'
            )
        return self.with_params(resolver.param_dict)

    @property
    def gate(self):
        circuit_qubits = ops.QubitOrder.DEFAULT.order_for(self.circuit.all_qubits())
        axis_map = {q: i for i, q in enumerate(circuit_qubits)}
        circuit_ops = tuple(self.circuit.all_operations())
        gateless_ops = [op for op in circuit_ops if op.gate is None]
        if gateless_ops:
            raise ValueError(f'Subcircuit contains gateless operations: {gateless_ops}.')
        gates = tuple((op.gate, tuple(axis_map[q] for q in op.qubits)) for op in circuit_ops)
        gate = CircuitGate(gates)  # type: cirq.Gate
        if self.measurement_key_map:
            gate = KeyMapGate(gate, self.measurement_key_map)
        if self.param_resolver:
            gate = ResolverGate(gate, self.param_resolver)
        if self.repetitions != 1:
            gate = RepeatGate(gate, self.repetitions)
        return gate


@dataclasses.dataclass(frozen=True)
class CircuitGate(ops.Gate):
    circuit: Tuple[Tuple[ops.Gate, Tuple[int, ...]], ...]

    def _qid_shape_(self):
        size = max(axis for _, axes in self.circuit for axis in axes) + 1
        dims = [-1] * size
        for g, axes in self.circuit:
            shape = protocols.qid_shape(g)
            for axis, dim in zip(axes, shape):
                if dims[axis] != -1 and dims[axis] != dim:
                    raise ValueError('ops of incompatible shapes')
                dims[axis] = dim
        if any(d == -1 for d in dims):
            raise ValueError('a gate is unused')
        return tuple(dims)

    def _has_unitary_(self):
        return all(protocols.has_unitary(g) for g, _ in self.circuit)

    def _is_measurement_(self) -> bool:
        return any(protocols.is_measurement(g) for g, _ in self.circuit)

    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        return set().union(*[protocols.measurement_key_objs(g) for g, _ in self.circuit])

    def _measurement_key_names_(self) -> AbstractSet[str]:
        return {str(key) for key in self._measurement_key_objs_()}

    def _parameter_names_(self) -> AbstractSet[str]:
        return set().union(*[protocols.parameter_names(g) for g, _ in self.circuit])

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return tuple(g.on(*[qubits[i] for i in axes]) for g, axes in self.circuit)

    def _act_on_(self, args: 'cirq.ActOnArgs', qubits: Sequence['cirq.Qid']) -> bool:
        for op in self._decompose_(qubits):
            protocols.act_on(op, args)
        return True

    def __pow__(self, power: int) -> 'CircuitGate':
        return CircuitGate(self.circuit * power)

    def _with_key_path_(self, path: Tuple[str, ...]):
        circuit = tuple((protocols.with_key_path(g, path), axes) for g, axes in self.circuit)
        return CircuitGate(circuit)

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        circuit = tuple(
            (protocols.with_key_path_prefix(g, prefix), axes) for g, axes in self.circuit
        )
        return CircuitGate(circuit)

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'CircuitGate':
        circuit = tuple(
            (protocols.with_measurement_key_mapping(g, key_map), axes) for g, axes in self.circuit
        )
        return CircuitGate(circuit)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'CircuitGate':
        circuit = tuple(
            (protocols.resolve_parameters(g, resolver, recursive), axes) for g, axes in self.circuit
        )
        return CircuitGate(circuit)


@dataclasses.dataclass(frozen=True)
class RepeatGate(ops.Gate):
    gate: ops.Gate
    repeat: int
    _path_prefix: Tuple[str] = dataclasses.field(default=tuple(), init=False)

    def _qid_shape_(self):
        return protocols.qid_shape(self.gate)

    def _has_unitary_(self):
        return protocols.has_unitary(self.gate)

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self.gate)

    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        return protocols.measurement_key_objs(self.gate)

    def _measurement_key_names_(self) -> AbstractSet[str]:
        return protocols.measurement_key_names(self.gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.gate)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        ops = protocols.decompose_once_with_qubits(self.gate, qubits)
        vals = []
        repeat = self.repeat
        if repeat < 0:
            repeat = -repeat
            ops = list(reversed(ops))
        for i in range(repeat):
            prefix = self._path_prefix + (str(i),)
            for op in ops:
                vals.append(protocols.with_key_path_prefix(op, prefix))
        return vals

    def __pow__(self, power: int) -> 'RepeatGate':
        return dataclasses.replace(self, repeat=self.repeat * power)

    def _with_key_path_(self, path: Tuple[str, ...]) -> 'RepeatGate':
        copy = dataclasses.replace(self)
        object.__setattr__(copy, '_path_prefix', path)
        return copy

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]) -> 'RepeatGate':
        copy = dataclasses.replace(self)
        object.__setattr__(copy, '_path_prefix', prefix + self._path_prefix)
        return copy

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'RepeatGate':
        return dataclasses.replace(
            self, gate=protocols.with_measurement_key_mapping(self.gate, key_map)
        )

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'RepeatGate':
        return dataclasses.replace(
            self, gate=protocols.resolve_parameters(self.gate, resolver, recursive)
        )


@dataclasses.dataclass(frozen=True)
class KeyMapGate(ops.Gate):
    gate: ops.Gate
    map: Dict[str, str]

    def _qid_shape_(self):
        return protocols.qid_shape(self.gate)

    def _has_unitary_(self):
        return protocols.has_unitary(self.gate)

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self.gate)

    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        return protocols.measurement_key_objs(
            protocols.with_measurement_key_mapping(self.gate, self.map)
        )

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(self.gate)

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return protocols.decompose_once_with_qubits(
            protocols.with_measurement_key_mapping(self.gate, self.map), qubits
        )

    def __pow__(self, power: int) -> 'KeyMapGate':
        return dataclasses.replace(self, gate=self.gate * power)

    def _with_key_path_(self, path: Tuple[str, ...]) -> 'KeyMapGate':
        return dataclasses.replace(self, gate=protocols.with_key_path(self.gate, path))

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]) -> 'KeyMapGate':
        return dataclasses.replace(self, gate=protocols.with_key_path_prefix(self.gate, prefix))

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'KeyMapGate':
        return KeyMapGate(self, key_map)

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'KeyMapGate':
        return dataclasses.replace(
            self, gate=protocols.resolve_parameters(self.gate, resolver, recursive)
        )


@dataclasses.dataclass(frozen=True)
class ResolverGate(ops.Gate):
    gate: ops.Gate
    map: 'cirq.ParamResolverOrSimilarType'

    def _qid_shape_(self):
        return protocols.qid_shape(self.gate)

    def _has_unitary_(self):
        return protocols.has_unitary(self.gate)

    def _is_measurement_(self) -> bool:
        return protocols.is_measurement(self.gate)

    def _measurement_key_objs_(self) -> AbstractSet[value.MeasurementKey]:
        return protocols.measurement_key_objs(self.gate)

    def _parameter_names_(self) -> AbstractSet[str]:
        return protocols.parameter_names(protocols.resolve_parameters(self.gate, self.map))

    def _decompose_(self, qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        return protocols.decompose_once_with_qubits(
            protocols.resolve_parameters(self.gate, self.map), qubits
        )

    def __pow__(self, power: int) -> 'ResolverGate':
        return dataclasses.replace(self, gate=self.gate * power)

    def _with_key_path_(self, path: Tuple[str, ...]) -> 'ResolverGate':
        return dataclasses.replace(self, gate=protocols.with_key_path(self.gate, path))

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]) -> 'ResolverGate':
        return dataclasses.replace(self, gate=protocols.with_key_path_prefix(self.gate, prefix))

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'ResolverGate':
        return dataclasses.replace(
            self, gate=protocols.with_measurement_key_mapping(self.gate, key_map)
        )

    def _resolve_parameters_(self, resolver: 'cirq.ParamResolver', recursive: bool) -> 'cirq.Gate':
        merged = protocols.resolve_parameters(resolver, self.map)
        if recursive:
            return protocols.resolve_parameters(self.gate, merged)
        return dataclasses.replace(self, map=merged)
