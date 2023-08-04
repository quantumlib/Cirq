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
from typing import Optional, List
from attr import frozen
from cirq._compat import cached_property
import cirq

from cirq_ft import infra


@frozen
class KitaevPhaseEstimation(infra.GateWithRegisters):
    r"""
    Performs phase estimation using Kitaev's method.

    This class represents the Kitaev Phase Estimation algorithm, which estimates the
    phase of an eigenstate of a unitary operator. It uses a register of qubits
    which are brought into a superposition, a series
    of controlled unitary operations, and another quantum Fourier transform.

    The algorithm assumes the existence of a unitary operator that can be applied in
    a controlled manner and that the eigenstates to be measured are available. The
    result is read from the controlled register after the inverse quantum Fourier transform.

    The controlled-U operation is applied conditionally based on the state of the
    first register (taken as an integer k), such that what is effectively happening is U^k is applied
    on the second register (initially the eigenstate |psi>).

    ASCII diagram of a generic phase estimation circuit:

    |0>   ----H----------------------------   QFT† ---- Measure
    |0>   ----H----------------------------   QFT† ---- Measure
    |0>   ----H----------------------------   QFT† ---- Measure
    .
    .
    |0>   ----H----------------------------   QFT† ---- Measure
    |psi> ------U^k--------------------------

    Here, |0> represents the initial state of each qubit in the controlled register,
    |psi> is the eigenstate of the unitary operator, H is the Hadamard gate, U^k
    is the controlled-U gate where U is the unitary operator and k is the state of the
    first register, and QFT† represents the inverse quantum Fourier transform applied
    only on the auxiliary register.

    Arguments:
    precision: int
        The number of bits of precision required in the phase estimation. This is the
        number of qubits in the controlled register.

    U: cirq.Gate
        The unitary operator whose eigenvalue phase is to be estimated. This should be a
        unitary quantum gate which conforms to `cirq.Gate`.

    References:
    Kitaev, A. Y. (1995). Quantum measurements and the Abelian Stabilizer Problem.
    Retrieved from https://arxiv.org/abs/quant-ph/9511026

    Nielsen, M. A., & Chuang, I. L. (2010). Quantum Computation and Quantum Information.
    Cambridge University Press. ISBN 978-1-107-00217-3
    """

    precision: int
    U: cirq.Gate

    @cached_property
    def registers(self) -> infra.Registers:
        if isinstance(self.U, infra.GateWithRegisters):
            return self.U.registers
        else:
            return infra.Registers(
                [infra.Register("phase_reg", self.precision), self.eigenvector_register]
            )

    @property
    @abc.abstractmethod
    def eigenvector_register(self) -> infra.Register:
        ...

    @cached_property
    def eigenvector_bitsize(self):
        return self.U.num_qubits()

    def qft_inverse(self, qubits):
        """Generator for the inverse QFT on a list of qubits."""
        qreg = list(qubits)
        while len(qreg) > 0:
            q_head = qreg.pop(0)
            yield cirq.H(q_head)
            for i, qubit in enumerate(qreg):
                yield (cirq.CZ ** (-1 / 2 ** (i + 1)))(qubit, q_head)

    def U_to_the_k_power(self, control_bits, eigen_vector_bits) -> List[cirq.Operation]:
        return [
            cirq.ControlledGate(self.U).on(bit, *eigen_vector_bits)
            ** (2 ** (self.precision - i - 1))
            for i, bit in enumerate(control_bits)
        ]

    def decompose_from_registers(
        self, context: cirq.DecompositionContext, **quregs
    ) -> cirq.OP_TREE:
        bits_of_precision = quregs["phase_reg"]
        eigenvector_bits = quregs[self.eigenvector_register.name]

        yield cirq.H.on_each(*bits_of_precision)
        yield [op for op in self.U_to_the_k_power(bits_of_precision, eigenvector_bits)]
        yield self.qft_inverse([*bits_of_precision])
