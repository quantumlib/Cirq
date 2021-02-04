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

from typing import TYPE_CHECKING, AbstractSet, Callable, Dict, List, Optional, Tuple, Union

import dataclasses
import numpy as np

from cirq import circuits, ops, protocols, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


INT_TYPE = Union[int, np.integer]


def default_repetition_ids(repetitions: int) -> List[str]:
    return [str(i) for i in range(abs(repetitions))]


def cartesian_product_of_string_lists(list1: List[str], list2: List[str]):
    return [f'{first}-{second}' for first in list1 for second in list2]


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
        param_resolver: Resolved values for parameters in the circuit.
        repetition_ids: List of identifiers for each repetition of the
            CircuitOperation. If populated, the length should be equal to the
            repetitions. If not populated, it is initialized to strings for
            numbers 0 to self.repetitions.
    """

    _hash: Optional[int] = dataclasses.field(default=None, init=False)

    circuit: 'cirq.FrozenCircuit'
    repetitions: int = 1
    qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = dataclasses.field(default_factory=dict)
    measurement_key_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    param_resolver: study.ParamResolver = study.ParamResolver()
    repetition_ids: List[str] = dataclasses.field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.circuit, circuits.FrozenCircuit):
            raise TypeError(f'Expected circuit of type FrozenCircuit, got: {type(self.circuit)!r}')

        # Ensure that the circuit is invertible if the repetitions are negative.
        if self.repetitions < 0:
            try:
                protocols.inverse(self.circuit.unfreeze())
            except TypeError:
                raise ValueError(f'repetitions are negative but the circuit is not invertible')

        # Initialize repetition_ids to default, if unspecified. Else, validate their length.
        loop_size = abs(self.repetitions)
        if not self.repetition_ids:
            object.__setattr__(self, 'repetition_ids', self._default_repetition_ids())
        elif len(self.repetition_ids) != loop_size:
            raise ValueError(
                f'Expected repetition_ids to be a list of length {loop_size}, '
                f'got: {self.repetition_ids}'
            )
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
        )

    # Methods for getting post-mapping properties of the contained circuit.

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Returns the qubits operated on by this object."""
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(self.circuit.all_qubits())
        return tuple(self.qubit_map.get(q, q) for q in ordered_qubits)

    def _default_repetition_ids(self) -> List[str]:
        return default_repetition_ids(self.repetitions)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple(q.dimension for q in self.qubits)

    def _measurement_keys_(self) -> AbstractSet[str]:
        return {
            self.measurement_key_map.get(key, key) for key in self.circuit.all_measurement_keys()
        }

    def _parameter_names_(self) -> AbstractSet[str]:
        return {
            name
            for symbol in protocols.parameter_symbols(self.circuit)
            for name in protocols.parameter_names(
                protocols.resolve_parameters(symbol, self.param_resolver, recursive=False)
            )
        }

    def _decompose_(self) -> 'cirq.OP_TREE':
        result = self.circuit.unfreeze()
        result = result.transform_qubits(lambda q: self.qubit_map.get(q, q))
        if self.repetitions < 0:
            result = result ** -1
        result = protocols.with_measurement_key_mapping(result, self.measurement_key_map)
        result = protocols.resolve_parameters(result, self.param_resolver, recursive=False)
        # repetition_ids don't need to be taken into account if the circuit has no measurements
        # or if it has repetition of size 1 with default repetition_ids.
        if self.repetition_ids == default_repetition_ids(1) or not protocols.is_measurement(result):
            return list(result.all_operations()) * abs(self.repetitions)
        # If it's a measurement circuit with repetitions/repetition_ids, prefix the repetition_ids
        # to measurements. Details at https://tinyurl.com/measurement-repeated-circuitop.
        ops = []  # type: List[cirq.Operation]
        for parent_id in self.repetition_ids:
            for op in result.all_operations():
                if protocols.is_measurement(op):
                    if isinstance(op, CircuitOperation):
                        # For a measurement CircuitOperation, prefix the current repetition_id to
                        # the children repetition_ids.
                        ops.append(
                            op.with_repetition_ids(
                                cartesian_product_of_string_lists([parent_id], op.repetition_ids)
                            )
                        )
                    else:
                        # For a non-CircuitOperation measurement, prefix the current repetition_id
                        # to the children measurement keys. Implemented by creating a mapping and
                        # using the with_measurement_key_mapping protocol.
                        ops.append(
                            protocols.with_measurement_key_mapping(
                                op,
                                key_map={
                                    key: f'{parent_id}-{key}'
                                    for key in protocols.measurement_keys(op)
                                },
                            )
                        )
                else:
                    ops.append(op)
        return ops

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
        if self.repetition_ids != self._default_repetition_ids():
            # Default repetition_ids need not be specified.
            args += f'repetition_ids={proper_repr(self.repetition_ids)},\n'
        indented_args = args.replace('\n', '\n    ')
        return f'cirq.CircuitOperation({indented_args[:-4]})'

    def __str__(self):
        # TODO: support out-of-line subcircuit definition in string format.
        header = self.circuit.serialization_key() + ':'
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(header) - 4] + [len(line) for line in msg_lines])
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
        if self.repetition_ids != self._default_repetition_ids():
            # Default repetition_ids need not be specified.
            args.append(f'repetition_ids={self.repetition_ids}')
        elif self.repetitions != 1:
            # Only add loops if we haven't added repetition_ids.
            args.append(f'loops={self.repetitions}')
        if not args:
            return f'{header}\n{circuit_msg}'
        return f'{header}\n{circuit_msg}({", ".join(args)})'

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
                        tuple(self.repetition_ids),
                    )
                ),
            )
        return self._hash

    def _json_dict_(self):
        return {
            'cirq_type': 'CircuitOperation',
            'circuit': self.circuit,
            'repetitions': self.repetitions,
            # JSON requires mappings to have keys of basic types.
            # Pairs must be sorted to ensure consistent serialization.
            'qubit_map': sorted(self.qubit_map.items()),
            'measurement_key_map': self.measurement_key_map,
            'param_resolver': self.param_resolver,
            'repetition_ids': self.repetition_ids,
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
        **kwargs,
    ):
        return (
            cls(circuit)
            .with_qubit_mapping(dict(qubit_map))
            .with_measurement_key_mapping(measurement_key_map)
            .with_params(param_resolver)
            .repeat(repetitions, repetition_ids)
        )

    # Methods for constructing a similar object with one field modified.

    def repeat(
        self,
        repetitions: Optional[INT_TYPE] = None,
        repetition_ids: Optional[List[str]] = None,
        override_all_ids: Optional[bool] = None,
    ) -> 'CircuitOperation':
        """Returns a copy of this operation repeated 'repetitions' times.
         Each repetition instance will be identified by a single repetition_id.

        Args:
            repetitions: Number of times this operation should repeat. This
                is multiplied with any pre-existing repetitions. If unset,
                it is defaulted to a sane value that would validate the
                `repetition_ids` and `override_all_ids` values.
                If `override_all_ids` is evaluated to False, defaults to the
                length of repetition_ids. If override_all_ids is evaluated to
                True (details below), defaults to
                `len(repetition_ids) / abs(self.repetitions)`
            repetition_ids: List of IDs, one for each repetition. The size
                should be equal to either `repetitions` or
                `repetitions*self.repetitions` based on the evaluated value of
                `override_all_ids` (details below).
                If unset, defaults to `range(required_size)` converted to str
            override_all_ids: If True, the `repetition_ids` are directly
                applied to the final operation without taking
                `self.repetition_ids` into account. Input `repetition_ids` size
                needs to be equal to the final repetitions.
                If False, the `repetition_ids` of the returned CircuitOperation
                are the cartesian product of the input `repetition_ids` with
                `self.repetition_ids`. Input `repetition_ids` size needs to be
                equal to the input `repetitions`.
                If unset, defaults to False unless either 1) the base operation
                is a single default repetition (i.e. no actual repetitions) or
                2) both the base repetition_ids and the current input
                `repetition_ids` are default/None (i.e no customization).


        Returns:
            A copy of this operation repeated `repetitions` times with the
            appropriate `repetition_ids` as described above.

        Raises:
            TypeError: `repetitions` is not an integer value.
            ValueError: Unexpected length of `repetition_ids`.
            ValueError: Both `repetitions` and `repetition_ids` are None.
        """
        base_is_default_rep = self.repetition_ids == self._default_repetition_ids()
        base_is_single_default_rep = self.repetitions == 1 and base_is_default_rep
        no_custom_repetition_ids = repetition_ids is None and base_is_default_rep
        if override_all_ids is None:
            override_all_ids = base_is_single_default_rep or no_custom_repetition_ids

        if repetitions is None:
            if repetition_ids is None:
                raise ValueError('At least one of repetitions and repetition_ids must be set')
            if override_all_ids:
                if len(repetition_ids) % abs(self.repetitions) != 0:
                    raise ValueError(
                        f'Length of repetition_ids must be a multiple of {abs(self.repetitions)}'
                    )
                repetitions = int(len(repetition_ids) / abs(self.repetitions))
            else:
                repetitions = len(repetition_ids)

        if not isinstance(repetitions, (int, np.integer)):
            raise TypeError('Only integer repetitions are allowed.')

        repetitions = int(repetitions)
        # The eventual number of repetitions of the returned CircuitOperation.
        final_repetitions = self.repetitions * repetitions
        expected_repetition_id_length = (
            abs(final_repetitions) if override_all_ids else abs(repetitions)
        )

        if repetition_ids is None:
            repetition_ids = default_repetition_ids(expected_repetition_id_length)
        elif len(repetition_ids) != abs(expected_repetition_id_length):
            raise ValueError(
                f'Expected repetition_ids={repetition_ids} length to be '
                f'{expected_repetition_id_length}'
            )

        if repetitions == 1 and repetition_ids == self.repetition_ids:
            # As CircuitOperation is immutable, this can safely return the original.
            return self

        if not override_all_ids:
            repetition_ids = cartesian_product_of_string_lists(repetition_ids, self.repetition_ids)

        return self.replace(repetitions=final_repetitions, repetition_ids=repetition_ids)

    def __pow__(self, power: int) -> 'CircuitOperation':
        return self.repeat(power)

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

        Returns:
            A copy of this operation with measurement keys updated as specified
                by key_map.

        Raises:
            ValueError: The new operation has a different number of measurement
                keys than this operation.
        """
        new_map = {}
        for k in self.circuit.all_measurement_keys():
            k_new = self.measurement_key_map.get(k, k)
            k_new = key_map.get(k_new, k_new)
            if k_new != k:
                new_map[k] = k_new
        new_op = self.replace(measurement_key_map=new_map)
        if len(new_op._measurement_keys_()) != len(self._measurement_keys_()):
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
        self, param_resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'CircuitOperation':
        if recursive:
            raise ValueError(
                'Recursive resolution of CircuitOperation parameters is prohibited. '
                'Use "recursive=False" to prevent this error.'
            )
        return self.with_params(param_resolver.param_dict)
