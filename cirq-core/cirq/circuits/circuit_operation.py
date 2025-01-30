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
import math
from functools import cached_property
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import numpy as np
import sympy

from cirq import circuits, ops, protocols, value, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


INT_CLASSES = (int, np.integer)
INT_TYPE = Union[int, np.integer]
IntParam = Union[INT_TYPE, sympy.Expr]
REPETITION_ID_SEPARATOR = '-'


def default_repetition_ids(repetitions: IntParam) -> Optional[List[str]]:
    if isinstance(repetitions, INT_CLASSES) and abs(repetitions) != 1:
        abs_repetitions: int = abs(int(repetitions))
        return [str(i) for i in range(abs_repetitions)]
    return None


def _full_join_string_lists(
    list1: Optional[Sequence[str]], list2: Optional[Sequence[str]]
) -> Optional[Sequence[str]]:
    if list1 is None and list2 is None:
        return None  # pragma: no cover
    if list1 is None:
        return list2  # pragma: no cover
    if list2 is None:
        return list1
    return [f'{first}{REPETITION_ID_SEPARATOR}{second}' for first in list1 for second in list2]


class CircuitOperation(ops.Operation):
    """An operation that encapsulates a circuit.

    This class captures modifications to the contained circuit, such as tags
    and loops, to support more condensed serialization. Similar to
    GateOperation, this type is immutable.
    """

    def __init__(
        self,
        circuit: 'cirq.FrozenCircuit',
        repetitions: INT_TYPE = 1,
        qubit_map: Optional[Dict['cirq.Qid', 'cirq.Qid']] = None,
        measurement_key_map: Optional[Dict[str, str]] = None,
        param_resolver: Optional[study.ParamResolverOrSimilarType] = None,
        repetition_ids: Optional[Sequence[str]] = None,
        parent_path: Tuple[str, ...] = (),
        extern_keys: FrozenSet['cirq.MeasurementKey'] = frozenset(),
        use_repetition_ids: Optional[bool] = None,
        repeat_until: Optional['cirq.Condition'] = None,
    ):
        """Initializes a CircuitOperation.

        Args:
            circuit: The FrozenCircuit wrapped by this operation.
            repetitions: How many times the circuit should be repeated. This can be
                integer, or a sympy expression. If sympy, the expression must
                resolve to an integer, or float within 0.001 of integer, at
                runtime.
            qubit_map: Remappings for qubits in the circuit.
            measurement_key_map: Remappings for measurement keys in the circuit.
                The keys and values should be unindexed (i.e. without repetition_ids).
                The values cannot contain the `MEASUREMENT_KEY_SEPARATOR`.
            param_resolver: Resolved values for parameters in the circuit.
            repetition_ids: List of identifiers for each repetition of the
                CircuitOperation. If populated, the length should be equal to the
                repetitions. If not populated and abs(`repetitions`) > 1, it is
                initialized to strings for numbers in `range(repetitions)`.
            parent_path: A tuple of identifiers for any parent CircuitOperations
                containing this one.
            extern_keys: The set of measurement keys defined at extern scope. The
                values here are used by decomposition and simulation routines to
                cache which external measurement keys exist as possible binding
                targets for unbound `ClassicallyControlledOperation` keys. This
                field is not intended to be set or changed manually, and should be
                empty in circuits that aren't in the middle of decomposition.
            use_repetition_ids: When True, any measurement key in the subcircuit
                will have its path prepended with the repetition id for each
                repetition. When False, this will not happen and the measurement
                key will be repeated. When None, default to False unless the caller
                passes `repetition_ids` explicitly.
            repeat_until: A condition that will be tested after each iteration of
                the subcircuit. The subcircuit will repeat until condition returns
                True, but will always run at least once, and the measurement key
                need not be defined prior to the subcircuit (but must be defined in
                a measurement within the subcircuit). This field is incompatible
                with repetitions or repetition_ids.

        Raises:
            TypeError: if repetitions is not an integer or sympy expression, or if
                the provided circuit is not a FrozenCircuit.
            ValueError: if any of the following conditions is met.
                - Negative repetitions on non-invertible circuit
                - Number of repetition IDs does not match repetitions
                - Repetition IDs used with parameterized repetitions
                - Conflicting qubit dimensions in qubit_map
                - Measurement key map has invalid key names
                - repeat_until used with other repetition controls
                - Key(s) in repeat_until are not modified by circuit
        """
        # This fields is exclusively for use in decomposition. It should not be
        # referenced outside this class.
        self._extern_keys = extern_keys

        # All other fields are pseudo-private: read access is allowed via the
        # @property methods, but mutation is prohibited.
        self._param_resolver = study.ParamResolver(param_resolver)
        self._parent_path = parent_path

        self._circuit = circuit
        if not isinstance(self._circuit, circuits.FrozenCircuit):
            raise TypeError(f'Expected circuit of type FrozenCircuit, got: {type(self._circuit)!r}')

        # Ensure that the circuit is invertible if the repetitions are negative.
        self._repetitions = repetitions
        self._repetition_ids = None if repetition_ids is None else list(repetition_ids)
        if use_repetition_ids is None:
            use_repetition_ids = repetition_ids is not None
        self._use_repetition_ids = use_repetition_ids
        if isinstance(self._repetitions, float):
            if math.isclose(self._repetitions, round(self._repetitions)):
                self._repetitions = round(self._repetitions)
        if isinstance(self._repetitions, INT_CLASSES):
            if self._repetitions < 0:
                try:
                    protocols.inverse(self._circuit.unfreeze())
                except TypeError:
                    raise ValueError('repetitions are negative but the circuit is not invertible')

            # Initialize repetition_ids to default, if unspecified. Else, validate their length.
            loop_size = abs(self._repetitions)
            if not self._repetition_ids:
                self._repetition_ids = self._default_repetition_ids()
            elif len(self._repetition_ids) != loop_size:
                raise ValueError(
                    f'Expected repetition_ids to be a list of length {loop_size}, '
                    f'got: {self._repetition_ids}'
                )
        elif isinstance(self._repetitions, sympy.Expr):
            if self._repetition_ids is not None:
                raise ValueError('Cannot use repetition ids with parameterized repetitions')
        else:
            raise TypeError(
                f'Only integer or sympy repetitions are allowed.\n'
                f'User provided: {self._repetitions}'
            )

        # Disallow qid mapping dimension conflicts.
        self._qubit_map = dict(qubit_map or {})
        for q, q_new in self._qubit_map.items():
            if q_new.dimension != q.dimension:
                raise ValueError(f'Qid dimension conflict.\nFrom qid: {q}\nTo qid: {q_new}')

        self._measurement_key_map = dict(measurement_key_map or {})
        # Disallow mapping to keys containing the `MEASUREMENT_KEY_SEPARATOR`
        for mapped_key in self._measurement_key_map.values():
            if value.MEASUREMENT_KEY_SEPARATOR in mapped_key:
                raise ValueError(
                    f'Mapping to invalid key: {mapped_key}. "{value.MEASUREMENT_KEY_SEPARATOR}" '
                    'is not allowed for measurement keys in a CircuitOperation'
                )

        self._repeat_until = repeat_until
        mapped_repeat_until = self._mapped_repeat_until
        if mapped_repeat_until:
            if self._use_repetition_ids or self._repetitions != 1:
                raise ValueError('Cannot use repetitions with repeat_until')
            if protocols.measurement_key_objs(self._mapped_single_loop()).isdisjoint(
                mapped_repeat_until.keys
            ):
                raise ValueError('Infinite loop: condition is not modified in subcircuit.')

    @property
    def circuit(self) -> 'cirq.FrozenCircuit':
        return self._circuit

    @property
    def repetitions(self) -> IntParam:
        return self._repetitions

    @property
    def repetition_ids(self) -> Optional[Sequence[str]]:
        return self._repetition_ids

    @property
    def use_repetition_ids(self) -> bool:
        return self._use_repetition_ids

    @property
    def repeat_until(self) -> Optional['cirq.Condition']:
        return self._repeat_until

    @property
    def qubit_map(self) -> Mapping['cirq.Qid', 'cirq.Qid']:
        return self._qubit_map

    @property
    def measurement_key_map(self) -> Mapping[str, str]:
        return self._measurement_key_map

    @property
    def param_resolver(self) -> study.ParamResolver:
        return self._param_resolver

    @property
    def parent_path(self) -> Tuple[str, ...]:
        return self._parent_path

    def base_operation(self) -> 'cirq.CircuitOperation':
        """Returns a copy of this operation with only the wrapped circuit.

        Key and qubit mappings, parameter values, and repetitions are not copied.
        """
        return CircuitOperation(self.circuit)

    def replace(self, **changes) -> 'cirq.CircuitOperation':
        """Returns a copy of this operation with the specified changes."""
        kwargs = {
            'circuit': self.circuit,
            'repetitions': self.repetitions,
            'qubit_map': self.qubit_map,
            'measurement_key_map': self.measurement_key_map,
            'param_resolver': self.param_resolver,
            'repetition_ids': self.repetition_ids,
            'parent_path': self.parent_path,
            'extern_keys': self._extern_keys,
            'use_repetition_ids': (
                True if changes.get('repetition_ids') is not None else self.use_repetition_ids
            ),
            'repeat_until': self.repeat_until,
            **changes,
        }
        return CircuitOperation(**kwargs)

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
            and self.use_repetition_ids == other.use_repetition_ids
            and self.repeat_until == other.repeat_until
        )

    # Methods for getting post-mapping properties of the contained circuit.

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Returns the qubits operated on by this object."""
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(self.circuit.all_qubits())
        return tuple(self.qubit_map.get(q, q) for q in ordered_qubits)

    def _default_repetition_ids(self) -> Optional[List[str]]:
        return default_repetition_ids(self.repetitions) if self.use_repetition_ids else None

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple(q.dimension for q in self.qubits)

    def _is_measurement_(self) -> bool:
        return self.circuit._is_measurement_()

    def _has_unitary_(self) -> bool:
        # Return false if parameterized for early exit of has_unitary protocol.
        # Otherwise return NotImplemented instructing the protocol to try alternate strategies
        if self._is_parameterized_() or self.repeat_until:
            return False
        return NotImplemented

    def _ensure_deterministic_loop_count(self):
        if self.repeat_until or isinstance(self.repetitions, sympy.Expr):
            raise ValueError('Cannot unroll circuit due to nondeterministic repetitions')

    @cached_property
    def _measurement_key_objs(self) -> FrozenSet['cirq.MeasurementKey']:
        circuit_keys = protocols.measurement_key_objs(self.circuit)
        if circuit_keys and self.use_repetition_ids:
            self._ensure_deterministic_loop_count()
            if self.repetition_ids is not None:
                circuit_keys = frozenset(
                    key.with_key_path_prefix(repetition_id)
                    for repetition_id in self.repetition_ids
                    for key in circuit_keys
                )
        circuit_keys = frozenset(
            key.with_key_path_prefix(*self.parent_path) for key in circuit_keys
        )
        return frozenset(
            protocols.with_measurement_key_mapping(key, self.measurement_key_map)
            for key in circuit_keys
        )

    def _measurement_key_objs_(self) -> FrozenSet['cirq.MeasurementKey']:
        return self._measurement_key_objs

    def _measurement_key_names_(self) -> FrozenSet[str]:
        return frozenset(str(key) for key in self._measurement_key_objs_())

    @cached_property
    def _control_keys(self) -> FrozenSet['cirq.MeasurementKey']:
        keys = (
            frozenset()
            if not protocols.control_keys(self.circuit)
            else protocols.control_keys(self._mapped_single_loop())
        )
        mapped_repeat_until = self._mapped_repeat_until
        if mapped_repeat_until is not None:
            keys |= frozenset(mapped_repeat_until.keys) - self._measurement_key_objs_()
        return keys

    def _control_keys_(self) -> FrozenSet['cirq.MeasurementKey']:
        return self._control_keys

    def _is_parameterized_(self) -> bool:
        return any(self._parameter_names_generator())

    def _parameter_names_(self) -> FrozenSet[str]:
        return frozenset(self._parameter_names_generator())

    def _parameter_names_generator(self) -> Iterator[str]:
        yield from protocols.parameter_names(self.repetitions)
        yield from protocols.parameter_names(self._mapped_repeat_until)
        yield from protocols.parameter_names(self._mapped_any_loop)

    @cached_property
    def _mapped_any_loop(self) -> 'cirq.Circuit':
        circuit = self.circuit.unfreeze()
        if self.qubit_map:
            circuit = circuit.transform_qubits(lambda q: self.qubit_map.get(q, q))
        if isinstance(self.repetitions, INT_CLASSES) and self.repetitions < 0:
            circuit = circuit**-1
        if self.measurement_key_map:
            circuit = protocols.with_measurement_key_mapping(circuit, self.measurement_key_map)
        if self.param_resolver:
            circuit = protocols.resolve_parameters(circuit, self.param_resolver, recursive=False)
        return circuit.unfreeze(copy=False)

    def _mapped_single_loop(self, repetition_id: Optional[str] = None) -> 'cirq.Circuit':
        circuit = self._mapped_any_loop
        if repetition_id:
            circuit = protocols.with_rescoped_keys(circuit, (repetition_id,))
        return protocols.with_rescoped_keys(
            circuit, self.parent_path, bindable_keys=self._extern_keys
        )

    @cached_property
    def _mapped_repeat_until(self) -> Optional['cirq.Condition']:
        """Applies measurement_key_map, param_resolver, and current scope to repeat_until."""
        repeat_until = self.repeat_until
        if not repeat_until:
            return repeat_until
        if self.measurement_key_map:
            repeat_until = protocols.with_measurement_key_mapping(
                repeat_until, self.measurement_key_map
            )
        if self.param_resolver:
            repeat_until = protocols.resolve_parameters(
                repeat_until, self.param_resolver, recursive=False
            )
        return protocols.with_rescoped_keys(
            repeat_until,
            self.parent_path,
            bindable_keys=self._extern_keys | self._measurement_key_objs,
        )

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
        self._ensure_deterministic_loop_count()
        if self.repetitions == 0:
            return circuits.Circuit()
        circuit = (
            circuits.Circuit(self._mapped_single_loop(rep) for rep in self.repetition_ids)
            if self.repetition_ids is not None
            and self.use_repetition_ids
            and protocols.is_measurement(self.circuit)
            else self._mapped_single_loop() * cast(IntParam, abs(self.repetitions))
        )
        if deep:
            circuit = circuit.map_operations(
                lambda op: op.mapped_circuit(deep=True) if isinstance(op, CircuitOperation) else op
            )
        return circuit

    def mapped_op(self, deep: bool = False) -> 'cirq.CircuitOperation':
        """As `mapped_circuit`, but wraps the result in a CircuitOperation."""
        return CircuitOperation(circuit=self.mapped_circuit(deep=deep).freeze())

    def _decompose_(self) -> Iterator['cirq.Operation']:
        return self.mapped_circuit(deep=False).all_operations()

    def _act_on_(self, sim_state: 'cirq.SimulationStateBase') -> bool:
        mapped_repeat_until = self._mapped_repeat_until
        if mapped_repeat_until:
            circuit = self._mapped_single_loop()
            while True:
                for op in circuit.all_operations():
                    protocols.act_on(op, sim_state)
                if mapped_repeat_until.resolve(sim_state.classical_data):
                    break
        else:
            for op in self._decompose_():
                protocols.act_on(op, sim_state)
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
        if self.use_repetition_ids:
            # Default repetition_ids need not be specified.
            args += f'repetition_ids={proper_repr(self.repetition_ids)},\n'
        if self.repeat_until:
            args += f'repeat_until={self.repeat_until!r},\n'
        indented_args = args.replace('\n', '\n    ')
        return f'cirq.CircuitOperation({indented_args[:-4]})'

    def __str__(self):
        # TODO: support out-of-line subcircuit definition in string format.
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(line) for line in msg_lines])
        circuit_msg = '\n'.join(f'[ {line:<{msg_width}} ]' for line in msg_lines)
        args = []

        def dict_str(d: Mapping) -> str:
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
        if self.use_repetition_ids:
            if self.repetition_ids != self._default_repetition_ids():
                args.append(f'repetition_ids={self.repetition_ids}')
            else:
                # Default repetition_ids need not be specified.
                args.append(f'loops={self.repetitions}, use_repetition_ids=True')
        elif self.repetitions != 1:
            # Add loops if not using repetition_ids.
            args.append(f'loops={self.repetitions}')
        if self.repeat_until:
            args.append(f'until={self.repeat_until}')
        if not args:
            return circuit_msg
        return f'{circuit_msg}({", ".join(args)})'

    @cached_property
    def _hash(self) -> int:
        return hash(
            (
                self.circuit,
                self.repetitions,
                frozenset(self.qubit_map.items()),
                frozenset(self.measurement_key_map.items()),
                self.param_resolver,
                self.parent_path,
                () if self.repetition_ids is None else tuple(self.repetition_ids),
                self.use_repetition_ids,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __getstate__(self) -> Dict[str, Any]:
        # clear cached hash value when pickling, see #6674
        state = self.__dict__
        # cached_property stores value in the property-named attribute
        hash_attr = "_hash"
        if hash_attr in state:
            state = state.copy()
            del state[hash_attr]
        return state

    def _json_dict_(self):
        resp = {
            'circuit': self.circuit,
            'repetitions': self.repetitions,
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'qubit_map': sorted(self.qubit_map.items()),
            'measurement_key_map': self.measurement_key_map,
            'param_resolver': self.param_resolver,
            'repetition_ids': self.repetition_ids,
            'use_repetition_ids': self.use_repetition_ids,
            'parent_path': self.parent_path,
        }
        if self.repeat_until:
            resp['repeat_until'] = self.repeat_until
        return resp

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
        use_repetition_ids=True,
        repeat_until=None,
        **kwargs,
    ):
        return cls(
            circuit=circuit,
            repetitions=repetitions,
            repetition_ids=repetition_ids,
            use_repetition_ids=use_repetition_ids,
            repeat_until=repeat_until,
            param_resolver=param_resolver,
            qubit_map=dict(qubit_map),
            measurement_key_map=measurement_key_map,
            parent_path=tuple(parent_path),
        )

    # Methods for constructing a similar object with one field modified.

    def repeat(
        self,
        repetitions: Optional[IntParam] = None,
        repetition_ids: Optional[Sequence[str]] = None,
        use_repetition_ids: Optional[bool] = None,
    ) -> 'CircuitOperation':
        """Returns a copy of this operation repeated 'repetitions' times.
         Each repetition instance will be identified by a single repetition_id.

        Args:
            repetitions: Number of times this operation should repeat. This
                is multiplied with any pre-existing repetitions. If unset, it
                defaults to the length of `repetition_ids`.
            repetition_ids: List of IDs, one for each repetition. If unset,
                defaults to `default_repetition_ids(repetitions)`.
            use_repetition_ids: If given, this specifies the value for `use_repetition_ids`
                of the resulting circuit operation. If not given, we enable ids if
                `repetition_ids` is not None, and otherwise fall back to
                `self.use_repetition_ids`.

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
        if use_repetition_ids is None:
            use_repetition_ids = True if repetition_ids is not None else self.use_repetition_ids

        if repetitions is None:
            if repetition_ids is None:
                raise ValueError('At least one of repetitions and repetition_ids must be set')
            repetitions = len(repetition_ids)

        if isinstance(repetitions, INT_CLASSES):
            if repetitions == 1 and repetition_ids is None:
                # As CircuitOperation is immutable, this can safely return the original.
                return self

            expected_repetition_id_length: int = np.abs(repetitions)

            if repetition_ids is None:
                if use_repetition_ids:
                    repetition_ids = default_repetition_ids(expected_repetition_id_length)
            elif len(repetition_ids) != expected_repetition_id_length:
                raise ValueError(
                    f'Expected repetition_ids={repetition_ids} length to be '
                    f'{expected_repetition_id_length}'
                )

        # If either self.repetition_ids or repetitions is None, it returns the other unchanged.
        repetition_ids = _full_join_string_lists(repetition_ids, self.repetition_ids)

        # The eventual number of repetitions of the returned CircuitOperation.
        final_repetitions = protocols.mul(self.repetitions, repetitions)
        return self.replace(
            repetitions=final_repetitions,
            repetition_ids=repetition_ids,
            use_repetition_ids=use_repetition_ids,
        )

    def __pow__(self, power: IntParam) -> 'cirq.CircuitOperation':
        return self.repeat(power)

    def _with_key_path_(self, path: Tuple[str, ...]):
        return self.replace(parent_path=path)

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]):
        return self.replace(parent_path=prefix + self.parent_path)

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet['cirq.MeasurementKey']
    ):
        # The following line prevents binding to measurement keys in previous repeated subcircuits
        # "just because their repetition ids matched". If we eventually decide to change that
        # requirement and allow binding across subcircuits (possibly conditionally upon the key or
        # the subcircuit having some 'allow_cross_circuit_binding' field set), this is the line to
        # change or remove.
        bindable_keys = frozenset(k for k in bindable_keys if len(k.path) <= len(path))
        bindable_keys |= {k.with_key_path_prefix(*path) for k in self._extern_keys}
        path += self.parent_path
        return self.replace(parent_path=path, extern_keys=bindable_keys)

    def with_key_path(self, path: Tuple[str, ...]):
        """Alias for `cirq.with_key_path(self, path)`.

        Args:
            path: Tuple of strings representing an alternate path to assign to the measurement
                keys in this `CircuitOperation`.

        Returns:
            A copy of this object with `parent_path=path`.
        """
        return self._with_key_path_(path)

    def with_repetition_ids(self, repetition_ids: List[str]) -> 'cirq.CircuitOperation':
        """Returns a copy of this `CircuitOperation` with the given repetition IDs.

        Args:
            repetition_ids: List of new repetition IDs to use. Must have length equal to the
                existing number of repetitions.

        Returns:
            A copy of this object with `repetition_ids=repetition_ids`.
        """
        return self.replace(repetition_ids=repetition_ids)

    def with_qubit_mapping(
        self, qubit_map: Union[Mapping['cirq.Qid', 'cirq.Qid'], Callable[['cirq.Qid'], 'cirq.Qid']]
    ) -> 'cirq.CircuitOperation':
        """Returns a copy of this operation with an updated qubit mapping.

        Users should pass either 'qubit_map' or 'transform' to this method.

        Args:
            qubit_map: A mapping of old qubits to new qubits. This map will be
                composed with any existing qubit mapping.

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
            transform = lambda q: qubit_map.get(q, q)
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

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'cirq.CircuitOperation':
        """Returns a copy of this operation with an updated qubit mapping.

        Args:
            *new_qubits: A list of qubits to target. Qubits in this list are
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

    def with_measurement_key_mapping(self, key_map: Mapping[str, str]) -> 'cirq.CircuitOperation':
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
        for k_obj in protocols.measurement_keys_touched(self.circuit):
            k = k_obj.name
            k_new = self.measurement_key_map.get(k, k)
            k_new = key_map.get(k_new, k_new)
            if k_new != k:
                new_map[k] = k_new
        new_op = self.replace(measurement_key_map=new_map)
        if len(protocols.measurement_keys_touched(new_op)) != len(
            protocols.measurement_keys_touched(self)
        ):
            raise ValueError(
                f'Collision in measurement key map composition. Original map:\n'
                f'{self.measurement_key_map}\nApplied changes: {key_map}'
            )
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> 'cirq.CircuitOperation':
        return self.with_measurement_key_mapping(key_map)

    def with_params(
        self, param_values: 'cirq.ParamResolverOrSimilarType', recursive: bool = False
    ) -> 'cirq.CircuitOperation':
        """Returns a copy of this operation with an updated ParamResolver.

        Any existing parameter mappings will have their values updated given
        the provided mapping, and any new parameters will be added to the
        ParamResolver.

        Note that any resulting parameter mappings with no corresponding
        parameter in the base circuit will be omitted. These parameters do not
        apply to the `repetitions` field if that is parameterized.

        Args:
            param_values: A map or ParamResolver able to convert old param
                values to new param values. This map will be composed with any
                existing ParamResolver via single-step resolution.
            recursive: If True, resolves parameter values recursively over the
                resolver; otherwise performs a single resolution step. This
                behavior applies only to the passed-in mapping, for the current
                application. Existing parameters are never resolved recursively
                because a->b and b->a needs to be a valid mapping.

        Returns:
            A copy of this operation with its ParamResolver updated as specified
                by param_values.
        """
        new_params = {}
        for k in protocols.parameter_symbols(self.circuit) | protocols.parameter_symbols(
            self.repeat_until
        ):
            v = self.param_resolver.value_of(k, recursive=False)
            v = protocols.resolve_parameters(v, param_values, recursive=recursive)
            if v != k:
                new_params[k] = v
        return self.replace(param_resolver=new_params)

    def _resolve_parameters_(
        self, resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'cirq.CircuitOperation':
        resolved = self.with_params(resolver.param_dict, recursive)
        # repetitions can resolve to a float, but this is ok since constructor converts to
        # nearby int.
        return resolved.replace(
            repetitions=resolver.value_of(cast('cirq.TParamVal', self.repetitions), recursive)
        )
