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
"""An immutable version of the Circuit data structure."""

from typing import TYPE_CHECKING, Callable, Sequence, Tuple

from cirq import devices, ops
from cirq.circuits import AbstractCircuit, Circuit
from cirq.circuits.insert_strategy import InsertStrategy

if TYPE_CHECKING:
    import cirq


class FrozenCircuit(AbstractCircuit):
    """An immutable version of the Circuit data structure.

    FrozenCircuits are immutable (and therefore hashable), but otherwise behave
    identically to regular Circuits. Conversion between the two is handled with
    the `freeze` and `unfreeze` methods from AbstractCircuit.
    """

    def __init__(self,
                 *contents: 'cirq.OP_TREE',
                 strategy: 'cirq.InsertStrategy' = InsertStrategy.EARLIEST,
                 device: 'cirq.Device' = devices.UNCONSTRAINED_DEVICE) -> None:
        base = Circuit(contents, strategy=strategy, device=device)
        self._moments = tuple(base.moments)
        self._device = base.device

        # These variables are memoized when first requested.
        self._num_qubits = None
        self._qid_shape = None
        self._all_qubits = None
        self._all_operations = None
        self._unitary = None
        self._has_measurements = None
        self._all_measurement_keys = None
        self._are_all_measurements_terminal = None

    @property
    def moments(self) -> Sequence['cirq.Moment']:
        return self._moments

    @property
    def device(self) -> devices.Device:
        return self._device

    def __hash__(self):
        return hash((self.moments, self.device))

    # Memoized methods for commonly-retrieved properties.

    def _num_qubits_(self):
        if self._num_qubits is None:
            self._num_qubits = len(self.all_qubits())
        return self._num_qubits

    def _qid_shape_(self):
        if self._qid_shape is None:
            self._qid_shape = super()._qid_shape_()
        return self._qid_shape

    def _unitary_(self):
        if self._unitary is None:
            self._unitary = super()._unitary_()
        return self._unitary

    def all_qubits(self):
        if self._all_qubits is None:
            self._all_qubits = super().all_qubits()
        return self._all_qubits

    def all_operations(self):
        if self._all_operations is None:
            self._all_operations = tuple(super().all_operations())
        return self._all_operations

    def has_measurements(self):
        if self._has_measurements is None:
            self._has_measurements = super().has_measurements()
        return self._has_measurements

    def all_measurement_keys(self):
        if self._all_measurement_keys is None:
            self._all_measurement_keys = super().all_measurement_keys()
        return self._all_measurement_keys

    def are_all_measurements_terminal(self):
        if self._are_all_measurements_terminal is None:
            self._are_all_measurements_terminal = super(
            ).are_all_measurements_terminal()
        return self._are_all_measurements_terminal

    # End of memoized methods.

    def __add__(self, other):
        return (self.unfreeze() + other).freeze()

    def __radd__(self, other):
        return (other + self.unfreeze()).freeze()

    # TODO: handle multiplication / powers differently?
    def __mul__(self, other):
        return (self.unfreeze() * other).freeze()

    def __rmul__(self, other):
        return (other * self.unfreeze()).freeze()

    def __pow__(self, other):
        try:
            return (self.unfreeze()**other).freeze()
        except:
            return NotImplemented

    # pylint: disable=function-redefined
    def __getitem__(self, key):
        if isinstance(key, slice):
            sliced_circuit = FrozenCircuit(device=self.device)
            sliced_circuit._moments = tuple(self._moments[key])
            return sliced_circuit
        if isinstance(key, tuple):
            if len(key) != 2:
                raise ValueError('If key is tuple, it must be a pair.')
            moment_idx, qubit_idx = key
            # qubit_idx - Qid or Iterable[Qid].
            selected_moments = self._moments[moment_idx]
            if isinstance(selected_moments, tuple):
                if isinstance(qubit_idx, ops.Qid):
                    qubit_idx = [qubit_idx]
                new_circuit = FrozenCircuit(device=self.device)
                new_circuit._moments = tuple(
                    moment[qubit_idx] for moment in selected_moments)
                return new_circuit

        try:
            return super().__getitem__(key)
        except TypeError:
            raise TypeError(
                '__getitem__ called with key not of type slice, int or tuple.')

    # pylint: enable=function-redefined

    def with_device(
            self,
            new_device: 'cirq.Device',
            qubit_mapping: Callable[['cirq.Qid'], 'cirq.Qid'] = lambda e: e,
    ) -> 'FrozenCircuit':
        return FrozenCircuit.freeze(self.unfreeze().with_device(
            new_device, qubit_mapping))

    def _resolve_parameters_(self, param_resolver: 'cirq.ParamResolver'
                            ) -> 'FrozenCircuit':
        return FrozenCircuit.freeze(
            self.unfreeze()._resolve_parameters_(param_resolver))
