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

from __future__ import annotations

from typing import (
    Any,
    cast,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Tuple,
    TYPE_CHECKING,
    Union,
)

from cirq import protocols, value
from cirq.ops import (
    dense_pauli_string as dps,
    measurement_gate,
    op_tree,
    pauli_gates,
    pauli_string_phasor,
    raw_types,
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
        observable: Union[cirq.BaseDensePauliString, Iterable[cirq.Pauli]],
        key: Union[str, cirq.MeasurementKey] = '',
    ) -> None:
        """Inits PauliMeasurementGate.

        Args:
            observable: Pauli observable to measure. Any `Iterable[cirq.Pauli]`
                is a valid Pauli observable (with a +1 coefficient by default).
                If you wish to measure pauli observables with coefficient -1,
                then pass a `cirq.DensePauliString` as observable.
            key: The string key of the measurement.

        Raises:
            ValueError: If the observable is empty.
        """
        if not observable:
            raise ValueError(f'Pauli observable {observable} is empty.')
        if not all(
            isinstance(p, pauli_gates.Pauli) for p in cast(Iterable['cirq.Gate'], observable)
        ):
            raise ValueError(f'Pauli observable {observable} must be Iterable[`cirq.Pauli`].')
        coefficient = (
            observable.coefficient if isinstance(observable, dps.BaseDensePauliString) else 1
        )
        if coefficient not in [+1, -1]:
            raise ValueError(
                f'`cirq.DensePauliString` observable {observable} must have coefficient +1/-1.'
            )

        self._observable = dps.DensePauliString(observable, coefficient=coefficient)
        self._mkey = (
            key if isinstance(key, value.MeasurementKey) else value.MeasurementKey(name=key)
        )

    @property
    def key(self) -> str:
        return str(self.mkey)

    @property
    def mkey(self) -> cirq.MeasurementKey:
        return self._mkey

    def _qid_shape_(self) -> Tuple[int, ...]:
        return (2,) * len(self._observable)

    def _has_unitary_(self) -> bool:
        return False

    def with_key(self, key: Union[str, cirq.MeasurementKey]) -> PauliMeasurementGate:
        """Creates a pauli measurement gate with a new key but otherwise identical."""
        if key == self.key:
            return self
        return PauliMeasurementGate(self._observable, key=key)

    def _with_key_path_(self, path: Tuple[str, ...]) -> PauliMeasurementGate:
        return self.with_key(self.mkey._with_key_path_(path))

    def _with_key_path_prefix_(self, prefix: Tuple[str, ...]) -> PauliMeasurementGate:
        return self.with_key(self.mkey._with_key_path_prefix_(prefix))

    def _with_rescoped_keys_(
        self, path: Tuple[str, ...], bindable_keys: FrozenSet[cirq.MeasurementKey]
    ) -> PauliMeasurementGate:
        return self.with_key(protocols.with_rescoped_keys(self.mkey, path, bindable_keys))

    def _with_measurement_key_mapping_(self, key_map: Mapping[str, str]) -> PauliMeasurementGate:
        return self.with_key(protocols.with_measurement_key_mapping(self.mkey, key_map))

    def with_observable(
        self, observable: Union[cirq.BaseDensePauliString, Iterable[cirq.Pauli]]
    ) -> PauliMeasurementGate:
        """Creates a pauli measurement gate with the new observable and same key."""
        if (
            observable
            if isinstance(observable, dps.BaseDensePauliString)
            else dps.DensePauliString(observable)
        ) == self._observable:
            return self
        return PauliMeasurementGate(observable, key=self.key)

    def _is_measurement_(self) -> bool:
        return True

    def _measurement_key_name_(self) -> str:
        return self.key

    def _measurement_key_obj_(self) -> cirq.MeasurementKey:
        return self.mkey

    def observable(self) -> cirq.DensePauliString:
        """Pauli observable which should be measured by the gate."""
        return self._observable

    def _decompose_(
        self, qubits: Tuple[cirq.Qid, ...]
    ) -> Iterator[protocols.decompose_protocol.DecomposeResult]:
        any_qubit = qubits[0]
        to_z_ops = op_tree.freeze_op_tree(self._observable.on(*qubits).to_z_basis_ops())
        xor_decomp = tuple(pauli_string_phasor.xor_nonlocal_decompose(qubits, any_qubit))
        yield to_z_ops
        yield xor_decomp
        yield measurement_gate.MeasurementGate(
            1, self.mkey, invert_mask=(self._observable.coefficient != 1,)
        ).on(any_qubit)
        yield protocols.inverse(xor_decomp)
        yield protocols.inverse(to_z_ops)

    def _circuit_diagram_info_(self, args: cirq.CircuitDiagramInfoArgs) -> cirq.CircuitDiagramInfo:
        coefficient = '' if self._observable.coefficient == 1 else '-'
        symbols = [
            f'M({"" if i else coefficient}{self._observable[i]})'
            for i in range(len(self._observable))
        ]

        # Mention the measurement key.
        label_map = args.label_map or {}
        if not args.known_qubits or self.key != _default_measurement_key(args.known_qubits):
            if self.key not in label_map:
                symbols[0] += f"('{self.key}')"
        if self.key in label_map:
            symbols += '@'

        return protocols.CircuitDiagramInfo(tuple(symbols))

    def _op_repr_(self, qubits: Sequence[cirq.Qid]) -> str:
        args = [repr(self._observable.on(*qubits))]
        if self.key != _default_measurement_key(qubits):
            args.append(f'key={self.mkey!r}')
        arg_list = ', '.join(args)
        return f'cirq.measure_single_paulistring({arg_list})'

    def __repr__(self) -> str:
        return f'cirq.PauliMeasurementGate({self._observable!r}, {self.mkey!r})'

    def _value_equality_values_(self) -> Any:
        return self.key, self._observable

    def _json_dict_(self) -> Dict[str, Any]:
        return {'observable': self._observable, 'key': self.key}

    @classmethod
    def _from_json_dict_(cls, observable, key, **kwargs) -> PauliMeasurementGate:
        return cls(observable=observable, key=value.MeasurementKey.parse_serialized(key))


def _default_measurement_key(qubits: Iterable[raw_types.Qid]) -> str:
    return ','.join(str(q) for q in qubits)
