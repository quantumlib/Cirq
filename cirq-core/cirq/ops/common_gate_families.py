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

"""Common Gate Families used in cirq-core"""

from typing import cast, Optional, Type, Union

from cirq.ops import gateset, raw_types, parallel_gate, eigen_gate
from cirq import protocols


class AnyUnitaryGateFamily(gateset.GateFamily):
    def __init__(self, num_qubits: Optional[int] = None) -> None:
        self._num_qubits = num_qubits
        if num_qubits:
            name = f'{num_qubits}-Qubit UnitaryGateFamily'
            description = f'Accepts any {num_qubits}-qubit unitary gate.'
        else:
            name = 'Any-Qubit UnitaryGateFamily'
            description = 'Accepts any unitary gate.'
        super().__init__(raw_types.Gate, name=name, description=description)

    def _predicate(self, g: raw_types.Gate) -> bool:
        return (
            self._num_qubits is None or protocols.num_qubits(g) == self._num_qubits
        ) and protocols.has_unitary(g)

    def __repr__(self) -> str:
        return f'cirq.AnyUnitaryGateFamily(num_qubits = {self._num_qubits})'


class AnyIntegerPowerGateFamily(gateset.GateFamily):
    def __init__(self, gate: Type[eigen_gate.EigenGate]) -> None:
        if not (isinstance(gate, type) and issubclass(gate, eigen_gate.EigenGate)):
            raise ValueError(f'{gate} must be a subclass of `cirq.EigenGate`.')
        super().__init__(
            gate,
            name=f'AnyIntegerPowerGateFamily: {gate}',
            description=f'Accepts any instance `g` of `{gate}` s.t. `g.exponent` is an integer.',
        )

    def _predicate(self, g: raw_types.Gate) -> bool:
        if protocols.is_parameterized(g) or not super()._predicate(g):
            return False
        exp = cast(eigen_gate.EigenGate, g).exponent  # for mypy
        return int(exp) == exp

    def __repr__(self) -> str:
        return f'cirq.AnyIntegerPowerGateFamily({self._gate_str()})'


class ParallelGateFamily(gateset.GateFamily):
    def __init__(
        self,
        gate: Union[Type[raw_types.Gate], raw_types.Gate],
        *,
        name: Optional[str] = None,
        description: Optional[str] = None,
        max_parallel_allowed=None,
    ) -> None:
        if isinstance(gate, parallel_gate.ParallelGate):
            gate = cast(parallel_gate.ParallelGate, gate).sub_gate
        self._max_parallel_allowed = max_parallel_allowed
        super().__init__(gate, name=name, description=description)

    def _max_parallel_str(self):
        return self._max_parallel_allowed if self._max_parallel_allowed is not None else 'INF'

    def _default_name(self) -> str:
        return f'{self._max_parallel_str()} Parallel ' + super()._default_name()

    def _default_description(self) -> str:
        check_type = r'g == {}' if isinstance(self.gate, raw_types.Gate) else r'isinstance(g, {})'
        return (
            f'Accepts\n'
            f'1. `cirq.Gate` instances `g` s.t. `{check_type.format(self._gate_str())}` OR\n'
            f'2. `cirq.ParallelGate` instance `g` s.t. `g.sub_gate` satisfies 1. and '
            f'`cirq.num_qubits(g) <= {self._max_parallel_str()}` OR\n'
            f'3. `cirq.Operation` instance `op` s.t. `op.gate` satisfies 1. or 2.'
        )

    def _predicate(self, gate: raw_types.Gate) -> bool:
        if (
            self._max_parallel_allowed is not None
            and protocols.num_qubits(gate) > self._max_parallel_allowed
        ):
            return False
        gate = gate.sub_gate if isinstance(gate, parallel_gate.ParallelGate) else gate
        return super()._predicate(gate)

    def __repr__(self) -> str:
        return (
            f'cirq.ParallelGateFamily(gate={self._gate_str(repr)},'
            f'name="{self.name}", '
            f'description=r\'\'\'' + self.description + '\'\'\','
            f'max_parallel_allowed="{self._max_parallel_allowed}")'
        )
