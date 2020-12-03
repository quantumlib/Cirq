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

from typing import TYPE_CHECKING, Dict, List, Tuple

import numpy as np

from cirq import linalg, ops, protocols, study
from cirq._compat import proper_repr

if TYPE_CHECKING:
    import cirq


class CircuitOperation(ops.Operation):
    """An operation that encapsulates a circuit.

    This class captures modifications to the contained circuit, such as tags
    and loops, to support more condensed serialization. Similar to
    GateOperation, this type is immutable.
    """

    def __init__(self, circuit: 'cirq.FrozenCircuit') -> None:
        """Constructs a gate that wraps a circuit.

        Args:
            circuit: The FrozenCircuit encapsulated by this operation.
        """
        self._circuit: 'cirq.FrozenCircuit' = circuit
        self._repetitions: int = 1
        self._qubit_map: Dict['cirq.Qid', 'cirq.Qid'] = {q: q for q in circuit.all_qubits()}
        self._measurement_key_map: Dict[str, str] = {k: k for k in circuit.all_measurement_keys()}
        self._param_resolver = study.ParamResolver(
            {p: p for p in protocols.parameter_symbols(circuit)}
        )

    def base_operation(self) -> 'CircuitOperation':
        """Returns a copy of this operation with only the wrapped circuit.

        Key mappings, parameter values, and repetitions are not copied.
        """
        return CircuitOperation(self._circuit)

    def copy(self) -> 'CircuitOperation':
        """Returns a copy of this operation with the same circuit object."""
        new_op = self.base_operation()
        new_op._qubit_map = self.qubit_map.copy()
        new_op._measurement_key_map = self.measurement_key_map.copy()
        new_op._param_resolver = self.param_resolver
        new_op._repetitions = self.repetitions
        return new_op

    @property
    def circuit(self) -> 'cirq.FrozenCircuit':
        """The FrozenCircuit wrapped by this operation."""
        return self._circuit

    def _num_qubits_(self) -> int:
        return len(self.circuit.all_qubits())

    @property
    def repetitions(self) -> int:
        """The number of times this operation will repeat its circuit.

        May be negative if the circuit should be inverted.
        """
        return self._repetitions

    @property
    def qubit_map(self) -> Dict['cirq.Qid', 'cirq.Qid']:
        """The deferred qubit mapping for this operation's circuit."""
        return self._qubit_map

    @property
    def measurement_key_map(self) -> Dict[str, str]:
        """The deferred measurement key mapping for this operation's circuit.

        The keys of this map correspond to measurement keys as they are defined
        in the "inner" circuit (self.circuit) while the values are the keys
        reported to the "outer" circuit (the circuit containing this operation).
        """
        return self._measurement_key_map

    @property
    def param_resolver(self) -> study.ParamResolver:
        """The deferred parameter values for this operation's circuit.

        The keys of this map correspond to parameters as they are defined in
        the "inner" circuit (self.circuit) while the values are parameters
        (or values) used when applying this operation as part of a larger
        circuit.
        """
        return self._param_resolver

    def __eq__(self, other) -> bool:
        if not isinstance(other, type(self)):
            return False
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
        return tuple(self._qubit_map.get(q, q) for q in ordered_qubits)

    def _qid_shape_(self) -> Tuple[int, ...]:
        return tuple(q.dimension for q in self.qubits)

    def _measurement_keys_(self):
        return {
            self.measurement_key_map.get(key, key) for key in self.circuit.all_measurement_keys()
        }

    def _parameter_names_(self):
        return {
            name
            for symbol in protocols.parameter_symbols(self.circuit)
            for name in protocols.parameter_names(
                protocols.resolve_parameters(symbol, self.param_resolver)
            )
        }

    def _decompose_(self) -> 'cirq.OP_TREE':
        result = self.circuit.unfreeze()
        result = result.transform_qubits(lambda q: self.qubit_map[q])
        if self.repetitions < 0:
            result = result ** -1
        result = protocols.with_measurement_key_mapping(result, self.measurement_key_map)
        result = protocols.resolve_parameters(result, self.param_resolver)

        return list(result.all_operations()) * abs(self.repetitions)

    # Methods for string representation of the operation.

    def __repr__(self):
        base_repr = f'cirq.CircuitOperation({self.circuit!r})'
        base_op = self.base_operation()

        def dict_repr(d: Dict) -> str:
            pairs = [f'    {proper_repr(k)}: {proper_repr(v)},' for k, v in sorted(d.items())]
            return '\n'.join(['{'] + pairs + ['}'])

        if self.qubit_map != base_op.qubit_map:
            base_repr += f'.with_qubit_mapping({dict_repr(self.qubit_map)})'
        if self.measurement_key_map != base_op.measurement_key_map:
            base_repr += (
                '.with_measurement_key_mapping(' + f'{dict_repr(self.measurement_key_map)})'
            )
        if self.param_resolver != base_op.param_resolver:
            base_repr += f'.with_params({proper_repr(self.param_resolver)})'
        if self.repetitions != 1:
            base_repr += f'.repeat({self.repetitions})'
        return base_repr

    def __str__(self):
        # TODO: support out-of-line subcircuit definition in string format.
        header = self.circuit.serialization_key()
        msg_lines = str(self.circuit).split('\n')
        msg_width = max([len(header) - 4] + [len(line) for line in msg_lines])
        circuit_msg = '\n'.join(
            ['[ {line:<{width}} ]'.format(line=line, width=msg_width) for line in msg_lines]
        )
        args = []
        base_op = self.base_operation()

        def dict_str(d: Dict) -> str:
            pairs = [f'{k}: {v}' for k, v in sorted(d.items())]
            return '{' + ', '.join(pairs) + '}'

        if self.qubit_map != base_op.qubit_map:
            args.append(f'qubit_map={dict_str(self.qubit_map)}')
        if self.measurement_key_map != base_op.measurement_key_map:
            args.append(f'key_map={dict_str(self.measurement_key_map)}')
        if self.param_resolver != base_op.param_resolver:
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

    def with_circuit(self, new_circuit: 'cirq.AbstractCircuit') -> 'CircuitOperation':
        """Returns a copy of this operation with the provided circuit.

        Key mappings, parameter values, and repetitions are preserved.
        """
        if protocols.is_measurement(new_circuit) and self._repetitions != 1:
            raise NotImplementedError('Measurements cannot be added to looped circuits.')
        new_op = self.copy()
        new_op._circuit = new_circuit.freeze()
        return new_op

    def repeat(
        self,
        repetitions: int,
        modulus: int = 0,
        allow_invert: bool = False,
        validate_modulus: bool = False,
    ) -> 'CircuitOperation':
        """Returns a copy of this operation repeated 'repetitions' times.

        If "modulus" is specified, overall repetitions will be mapped to the
        range (-modulus/2, modulus/2] if "allow_invert" is true; otherwise,
        it will map to [0, modulus). If "validate_modulus" is also true, this
        will also check that the provided modulus is valid for this circuit.
        """
        if not isinstance(repetitions, int):
            raise TypeError('Only integer repetitions are allowed.')
        if protocols.is_measurement(self.circuit):
            raise NotImplementedError('Loops over measurements are not supported.')
        new_op = self.copy()
        new_op._repetitions *= repetitions
        if modulus:
            if validate_modulus and not linalg.allclose_up_to_global_phase(
                protocols.unitary(self.circuit * modulus), np.eye(np.prod(self.circuit.qid_shape()))
            ):
                raise ValueError('Raising the circuit to "modulus" ' 'must produce the identity.')
            # Map repetitions to (-exp_modulus / 2, exp_modulus / 2]
            new_op._repetitions %= modulus
            if allow_invert and new_op._repetitions > modulus // 2:
                new_op._repetitions -= modulus
        return new_op

    def __pow__(self, power: int) -> 'CircuitOperation':
        return self.repeat(power)

    def with_qubit_mapping(self, qubit_map: Dict['cirq.Qid', 'cirq.Qid']) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated qubit mapping.

        The provided key_map is composed with the existing map, so calling this
        function with qubit_map={GridQubit(0, 1): GridQubit(0, 2)} on a
        CircuitOperation with qubit map {GridQubit(1, 1): GridQubit(0, 1)} will
        result in a final map of {GridQubit(1, 1): GridQubit(0, 2)}.
        """
        new_op = self.copy()
        new_op._qubit_map = {k: qubit_map.get(v, v) for k, v in new_op.qubit_map.items()}
        return new_op

    def with_qubits(self, *new_qubits: 'cirq.Qid'):
        if len(new_qubits) != protocols.num_qubits(self.circuit):
            expected = protocols.num_qubits(self.circuit)
            raise ValueError(f'Expected {expected} qubits, got {len(new_qubits)}.')
        return self.with_qubit_mapping(dict(zip(self.qubits, new_qubits)))

    def with_measurement_key_mapping(self, key_map: Dict[str, str]) -> 'CircuitOperation':
        """Returns a copy of this operation with an updated key mapping.

        The provided key_map is composed with the existing map, so calling this
        function with key_map={'b': 'c'} on a CircuitOperation with key map
        {'a': 'b'} will result in a final map of {'a': 'c'}.
        """
        new_op = self.copy()
        new_op._measurement_key_map = {
            k: key_map.get(v, v) for k, v in new_op._measurement_key_map.items()
        }
        return new_op

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'CircuitOperation':
        return self.with_measurement_key_mapping(key_map)

    def with_params(
        self, param_values: study.ParamResolverOrSimilarType, recursive: bool = False
    ) -> 'CircuitOperation':
        """Returns a copy of this operation with updated param values.

        The provided param_values are composed with the existing values, so
        calling this function with param_values={'b': 1} on a CircuitOperation
        with map {'a': 'b'} will result in a final map of {'a': 1}.
        """
        new_op = self.copy()
        resolver = study.ParamResolver(param_values)
        new_op._param_resolver = protocols.resolve_parameters(
            self.param_resolver, resolver, recursive
        )
        return new_op

    def _resolve_parameters_(
        self, param_resolver: 'cirq.ParamResolver', recursive: bool
    ) -> 'CircuitOperation':
        return self.with_params(param_resolver.param_dict, recursive)
