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

"""An immutable version of the Circuit data structure with unassigned qubits.

FrozenCircuits contain a tuple of pseudo-Moments, each of which contains a set
of Gate + qubit pairs. When the FrozenCircuit is applied to a set of qubits,
its Gates will be assigned to the qubits as defined by the user.
"""

import abc
import numpy as np

from typing import Any, Callable, Iterable, Sequence, Tuple, overload

from cirq import devices, ops
from cirq.circuits import AbstractCircuit, Circuit
from cirq.circuits.insert_strategy import InsertStrategy




class FrozenCircuit(AbstractCircuit):

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

    @property
    def moments(self) -> Sequence['cirq.Moment']:
        return self._moments
    
    @property
    def device(self) -> devices.Device:
        return self._device

    @staticmethod
    def freeze(circuit: Circuit) -> 'FrozenCircuit':
        return FrozenCircuit(circuit.moments, device=circuit.device)

    def base_circuit(self) -> Circuit:
        return Circuit(self.moments, device=self.device)

    def __repr__(self):
        return super().__repr__().replace('AbstractCircuit', 'FrozenCircuit')

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        if cycle:
            p.text('FrozenCircuit(...)')
        else:
            super()._repr_pretty_(p, False)

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

    def all_qubits(self):
        if self._all_qubits is None:
            self._all_qubits = super().all_qubits()
        return self._all_qubits

    def all_operations(self):
        if self._all_operations is None:
            self._all_operations = tuple(super().all_operations())
        return self._all_operations

    # End of memoized methods.

    def __add__(self, other):
        return FrozenCircuit.freeze(self.base_circuit() + other)

    def __radd__(self, other):
        return FrozenCircuit.freeze(other + self.base_circuit())

    # TODO: handle multiplication / powers differently?
    def __mul__(self, other):
        return FrozenCircuit.freeze(self.base_circuit() * other)

    def __rmul__(self, other):
        return FrozenCircuit.freeze(other * self.base_circuit())

    def __pow__(self, other):
        return FrozenCircuit.freeze(self.base_circuit() ** other)

    # pylint: disable=function-redefined
    @overload
    def __getitem__(self, key: slice) -> 'cirq.FrozenCircuit':
        pass

    @overload
    def __getitem__(self,
                    key: Tuple[slice, 'cirq.Qid']) -> 'cirq.FrozenCircuit':
        pass

    @overload
    def __getitem__(self,
                    key: Tuple[slice, Iterable['cirq.Qid']]
                    ) -> 'cirq.FrozenCircuit':
        pass

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
                    moment[qubit_idx] for moment in selected_moments
                )
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
        return FrozenCircuit.freeze(
            self.base_circuit().with_device(new_device, qubit_mapping))
