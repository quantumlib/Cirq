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

from typing import Any, Dict, Iterable, Tuple, Sequence, TYPE_CHECKING, Union


from cirq import protocols, value
from cirq.ops import (
    raw_types,
    measurement_gate,
    op_tree,
    dense_pauli_string,
    pauli_gates,
    pauli_string_phasor,
)

if TYPE_CHECKING:
    import cirq


@value.value_equality
class PauliMeasurementGate(raw_types.Gate):
    """A gate that measures a Pauli observable.

    PauliMeasurementGate contains a key used to identify results of measurement
    and a list of Paulis which denote the observable to be measured.
    """

    def __init__(
        self,
        observable: Iterable['cirq.Pauli'],
        key: Union[str, value.MeasurementKey] = '',
    ) -> None:
        """Inits PauliMeasurementGate.

        Args:
            observable: Pauli observable to measure. Any `Iterable[cirq.Pauli]`
                is a valid Pauli observable, including `cirq.DensePauliString`
                instances, which do not contain any identity gates.
            key: The string key of the measurement.

        Raises:
            ValueError: If the observable is empty.
        """
        if not observable:
            raise ValueError(f'Pauli observable {observable} is empty.')
        if not all(isinstance(p, pauli_gates.Pauli) for p in observable):
            raise ValueError(f'Pauli observable {observable} must be Iterable[`cirq.Pauli`].')
        self._observable = tuple(observable)
        self.key = key  # type: ignore

    @property
    def key(self) -> str:
        return str(self.mkey)

    @key.setter
    def key(self, key: Union[str, value.MeasurementKey]) -> None:
        if isinstance(key, str):
            key = value.MeasurementKey(name=key)
        self.mkey = key

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * len(self._observable)

    def with_key(self, key: Union[str, value.MeasurementKey]) -> 'PauliMeasurementGate':
        """Creates a pauli measurement gate with a new key but otherwise identical."""
        if key == self.key:
            return self
        return PauliMeasurementGate(self._observable, key=key)

    def _with_key_path_(self, path: Tuple[str, ...]) -> 'PauliMeasurementGate':
        return self.with_key(self.mkey._with_key_path_(path))

    def _with_measurement_key_mapping_(self, key_map: Dict[str, str]) -> 'PauliMeasurementGate':
        return self.with_key(protocols.with_measurement_key_mapping(self.mkey, key_map))

    def with_observable(self, observable: Iterable['cirq.Pauli']) -> 'PauliMeasurementGate':
        """Creates a pauli measurement gate with the new observable and same key."""
        if tuple(observable) == self._observable:
            return self
        return PauliMeasurementGate(observable, key=self.key)

    def _is_measurement_(self) -> bool:
        return True

    def _measurement_key_name_(self) -> str:
        return self.key

    def _measurement_key_obj_(self) -> value.MeasurementKey:
        return self.mkey

    def observable(self) -> 'cirq.DensePauliString':
        """Pauli observable which should be measured by the gate."""
        return dense_pauli_string.DensePauliString(self._observable)

    def _decompose_(
        self, qubits: Tuple['cirq.Qid', ...]
    ) -> 'protocols.decompose_protocol.DecomposeResult':
        any_qubit = qubits[0]
        to_z_ops = op_tree.freeze_op_tree(self.observable().on(*qubits).to_z_basis_ops())
        xor_decomp = tuple(pauli_string_phasor.xor_nonlocal_decompose(qubits, any_qubit))
        yield to_z_ops
        yield xor_decomp
        yield measurement_gate.MeasurementGate(1, self.mkey).on(any_qubit)
        yield protocols.inverse(xor_decomp)
        yield protocols.inverse(to_z_ops)

    def _circuit_diagram_info_(
        self, args: 'cirq.CircuitDiagramInfoArgs'
    ) -> 'cirq.CircuitDiagramInfo':
        symbols = [f'M({g})' for g in self._observable]

        # Mention the measurement key.
        if not args.known_qubits or self.key != _default_measurement_key(args.known_qubits):
            symbols[0] += f"('{self.key}')"

        return protocols.CircuitDiagramInfo(tuple(symbols))

    def _op_repr_(self, qubits: Sequence['cirq.Qid']) -> str:
        args = [repr(self.observable().on(*qubits))]
        if self.key != _default_measurement_key(qubits):
            args.append(f'key={self.mkey!r}')
        arg_list = ', '.join(args)
        return f'cirq.measure_single_paulistring({arg_list})'

    def __repr__(self) -> str:
        return f'cirq.PauliMeasurementGate(' f'{self._observable!r}, ' f'{self.mkey!r})'

    def _value_equality_values_(self) -> Any:
        return self.key, self._observable

    def _json_dict_(self) -> Dict[str, Any]:
        return {
            'cirq_type': self.__class__.__name__,
            'observable': self._observable,
            'key': self.key,
        }

    @classmethod
    def _from_json_dict_(cls, observable, key, **kwargs) -> 'PauliMeasurementGate':
        return cls(
            observable=observable,
            key=value.MeasurementKey.parse_serialized(key),
        )


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)
