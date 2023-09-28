# Copyright 2023 The Cirq Developers
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

from typing import Any, Dict
from cirq import ops, value


@value.value_equality
class InternalGate(ops.Gate):
    """InternalGate is a placeholder gate for internal gates.

    InternalGate holds the information required to instantiate
    a gate of type `self.gate_name` with the arguments for the gate
    constructor stored in `self.gate_args`.
    """

    def __init__(
        self,
        gate_name: str,
        gate_module: str = 'pyle.cirqtools.pyle_gates',
        num_qubits: int = 1,
        **kwargs,
    ):
        """Instatiates an InternalGate.

        Arguments:
            gate_name: Gate class name.
            gate_module: The module of the gate (defualt: pyle.cirqtools.pyle_gates).
            num_qubits: Number of qubits that the gate acts on.
            **kwargs: The named arguments to be passed to the gate constructor.
        """
        self.gate_module = gate_module
        self.gate_name = gate_name
        self._num_qubits = num_qubits
        self.gate_args = kwargs

    def _num_qubits_(self) -> int:
        return self._num_qubits

    def __str__(self):
        gate_args = ', '.join(f'{k}={v}' for k, v in self.gate_args.items())
        return f'{self.gate_module}.{self.gate_name}({gate_args})'

    def __repr__(self) -> str:
        gate_args = ', '.join(f'{k}={repr(v)}' for k, v in self.gate_args.items())
        if gate_args != '':
            gate_args = ', ' + gate_args
        return (
            f"cirq_google.InternalGate(gate_name='{self.gate_name}', "
            f"gate_module='{self.gate_module}', "
            f"num_qubits={self._num_qubits}"
            f"{gate_args})"
        )

    def _json_dict_(self) -> Dict[str, Any]:
        return dict(
            gate_name=self.gate_name,
            gate_module=self.gate_module,
            num_qubits=self._num_qubits,
            **self.gate_args,
        )

    def _value_equality_values_(self):
        hashable = True
        for arg in self.gate_args.values():
            try:
                hash(arg)
            except TypeError:
                hashable = False
        return (
            self.gate_module,
            self.gate_name,
            self._num_qubits,
            frozenset(self.gate_args.items()) if hashable else self.gate_args,
        )
