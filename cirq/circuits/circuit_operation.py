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

from cirq import linalg, ops, protocols, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


INT_TYPE = Union[int, np.integer]


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
    """

    circuit: 'cirq.FrozenCircuit'
    repetitions: int = 1
    qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = dataclasses.field(default_factory=dict)
    measurement_key_map: Dict[str, str] = dataclasses.field(default_factory=dict)
    param_resolver: study.ParamResolver = study.ParamResolver()

    def __post_init__(self):
        # Ensure that param_resolver is converted to an actual ParamResolver.
        object.__setattr__(self, 'param_resolver', study.ParamResolver(self.param_resolver))

    def base_operation(self) -> 'CircuitOperation':
        """Returns a copy of this operation with only the wrapped circuit.

        Key and qubit mappings, parameter values, and repetitions are not copied.
        """
        return CircuitOperation(self.circuit)

    def updated_copy(
        self,
        *,
        circuit: Optional['cirq.FrozenCircuit'] = None,
        repetitions: Optional[int] = None,
        qubit_map: Optional[Dict['cirq.Qid', 'cirq.Qid']] = None,
        measurement_key_map: Optional[Dict[str, str]] = None,
        param_resolver: Optional[study.ParamResolver] = None,
    ):
        """Returns a copy of this operation with the specified fields updated."""
        return CircuitOperation(
            circuit=self.circuit if circuit is None else circuit,
            repetitions=self.repetitions if repetitions is None else repetitions,
            qubit_map=self.qubit_map if qubit_map is None else qubit_map,
            measurement_key_map=self.measurement_key_map
            if measurement_key_map is None
            else measurement_key_map,
            param_resolver=self.param_resolver if param_resolver is None else param_resolver,
        )

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented
        return (
            self.circuit == other.circuit
            and self.qubit_map == other.qubit_map
            and self.measurement_key_map == other.measurement_key_map
            and self.param_resolver == other.param_resolver
            and self.repetitions == other.repetitions
        )

    # Methods for getting post-mapping properties of the contained circuit.

    @property
    def qubits(self) -> Tuple['cirq.Qid', ...]:
        """Returns the qubits operated on by this object."""
        ordered_qubits = ops.QubitOrder.DEFAULT.order_for(self.circuit.all_qubits())
        return tuple(self.qubit_map.get(q, q) for q in ordered_qubits)

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
                protocols.resolve_parameters(symbol, self.param_resolver)
            )
        }

    def _decompose_(self) -> 'cirq.OP_TREE':
        result = self.circuit.unfreeze()
        result = result.transform_qubits(lambda q: self.qubit_map.get(q, q))
        if self.repetitions < 0:
            result = result ** -1
        result = protocols.with_measurement_key_mapping(result, self.measurement_key_map)
        result = protocols.resolve_parameters(result, self.param_resolver)

        return list(result.all_operations()) * abs(self.repetitions)

    # Methods for string representation of the operation.

    def __repr__(self):
        args = [f'circuit={self.circuit!r}']
        if self.repetitions != 1:
            args.append(f'repetitions={self.repetitions}')
        if self.qubit_map:
            args.append(f'qubit_map={proper_repr(self.qubit_map)}')
        if self.measurement_key_map:
            args.append(f'measurement_key_map={proper_repr(self.measurement_key_map)}')
        if self.param_resolver:
            args.append(f'param_resolver={proper_repr(self.param_resolver)}')
        argstr = ",\n".join(args)  # no backslashes allowed in fstring braces
        return f'cirq.CircuitOperation({argstr})'

    def __str__(self):
        # TODO: support out-of-line subcircuit definition in string format.
        header = self.circuit.serialization_key() + ':'
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(header) - 4] + [len(line) for line in msg_lines])
        circuit_msg = '\n'.join(
            ['[ {line:<{width}} ]'.format(line=line, width=msg_width) for line in msg_lines]
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
        if self.repetitions != 1:
            args.append(f'loops={self.repetitions}')
        if not args:
            return f'{header}\n{circuit_msg}'
        return f'{header}\n{circuit_msg}({", ".join(args)})'

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
        }

    @classmethod
    def _from_json_dict_(
        cls, circuit, repetitions, qubit_map, measurement_key_map, param_resolver, **kwargs
    ):
        result = (
            cls(circuit)
            .with_qubit_mapping(dict(qubit_map))
            .with_measurement_key_mapping(measurement_key_map)
            .with_params(param_resolver)
        )
        if repetitions != 1:
            return result.repeat(repetitions)
        return result

    # Methods for constructing a similar object with one field modified.

    def repeat(
        self,
        repetitions: INT_TYPE,
    ) -> 'CircuitOperation':
        """Returns a copy of this operation repeated 'repetitions' times.

        Args:
            repetitions: Number of times this operation should repeat. This
                is multiplied with any pre-existing repetitions.

        Returns:
            A copy of this operation repeated 'repetitions' times.

        Raises:
            TypeError: `repetitions` is not an integer value.
            NotImplementedError: The operation contains measurements and
                cannot have repetitions.
        """
        if not isinstance(repetitions, (int, np.integer)):
            raise TypeError('Only integer repetitions are allowed.')
        repetitions = int(repetitions)
        if protocols.is_measurement(self.circuit):
            raise NotImplementedError('Loops over measurements are not supported.')
        return self.updated_copy(repetitions=self.repetitions * repetitions)

    def __pow__(self, power: int) -> 'CircuitOperation':
        return self.repeat(power)

    def with_qubit_mapping(
        self,
        qubit_map: Optional[Dict['cirq.Qid', 'cirq.Qid']] = None,
        transform: Optional[Callable[['cirq.Qid'], 'cirq.Qid']] = None,
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
            ValueError: The new operation has a different number of qubits than
                this operation, or the wrong number of arguments were provided
                (should be exactly one).
        """
        if transform is None:
            if qubit_map is None:
                raise ValueError('with_qubit_mapping requires one of {qubit_map, transform}.')
            else:
                transform = lambda q: qubit_map.get(q, q)  # type: ignore
        elif qubit_map is not None:
            raise ValueError('with_qubit_mapping received both of {qubit_map, transform}.')
        base_map = {q: q for q in self.circuit.all_qubits()}
        base_map.update(self.qubit_map)
        new_map = {}
        for k, v in base_map.items():
            new_v = transform(v)
            if k != new_v:
                new_map[k] = new_v
        new_op = self.updated_copy(qubit_map=new_map)
        if len(set(new_op.qubits)) != len(set(self.qubits)):
            raise ValueError(
                f'Collision in qubit map composition. Original map:\n{self.qubit_map}'
                f'\nMap after changes: {new_op.qubit_map}'
            )
        return new_op

    def with_qubits(self, *new_qubits: 'cirq.Qid'):
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
        base_map = {m: m for m in self.circuit.all_measurement_keys()}
        base_map.update(self.measurement_key_map)
        new_map = {k: key_map.get(v, v) for k, v in base_map.items() if k != key_map.get(v, v)}
        new_op = self.updated_copy(measurement_key_map=new_map)
        if len(new_op._measurement_keys_()) != len(self._measurement_keys_()):
            raise ValueError(
                f'Collision in measurement key map composition. Original map:\n'
                f'{self.measurement_key_map}\nApplied changes: {key_map}'
            )
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'CircuitOperation':
        return self.with_measurement_key_mapping(key_map)

    def with_params(
        self, param_values: study.ParamResolverOrSimilarType, recursive: bool = False
    ) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated ParamResolver.

        Args:
            param_values: A map or ParamResolver able to convert old param
                values to new param values. This map will be composed with any
                existing ParamResolver.

        Returns:
            A copy of this operation with its ParamResolver updated as specified
                by param_values.
        """
        new_resolver = protocols.resolve_parameters(self.param_resolver, param_values, recursive)
        return self.updated_copy(param_resolver=new_resolver)

    def _resolve_parameters_(
        self, param_resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'CircuitOperation':
        return self.with_params(param_resolver.param_dict, recursive)
