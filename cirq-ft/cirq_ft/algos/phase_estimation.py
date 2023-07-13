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
import abc
from cirq._compat import cached_property
import cirq

from cirq_ft import infra


class PhaseEstimation(infra.GateWithRegisters):
    r"""Abstract base class that defines the API for a Phase Estimation procedure"""

    @property
    @abc.abstractmethod
    def control_register(self) -> infra.Register:
        ...

    @property
    @abc.abstractmethod
    def eigenvector_register(self) -> infra.Register:
        ...

    @property
    @abc.abstractmethod
    def control_register_prep(self) -> infra.GateWithRegisters:
        ...

    @property
    @abc.abstractmethod
    def eigen_vector_register_prep(self) -> cirq.Gate:
        ...

    @cached_property
    def control_register_name(self):
        return self.control_register.name

    @cached_property
    def eigenvector_register_name(self):
        return self.eigenvector_register.name

    @property
    @abc.abstractmethod
    def lamda(self) -> infra.GateWithRegisters:
        ...

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([self.control_register, self.eigenvector_register])

    def qft_inverse(self, qubits):
        """Generator for the inverse QFT on a list of qubits."""
        qreg = list(qubits)[::-1]
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qreg):
                yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        control_register_qubits = quregs[self.control_register_name]
        eigenvector_register_qubits = quregs[self.eigenvector_register_name]
        yield self.eigen_vector_register_prep.on(*eigenvector_register_qubits)
        yield self.control_register_prep.on(*control_register_qubits)
        yield self.lamda.on_registers(
            control_register=[*control_register_qubits],
            eigenvector_register=[*eigenvector_register_qubits],
        )
        yield self.qft_inverse([*control_register_qubits][::-1])
