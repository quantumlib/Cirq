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
from typing import Tuple

from cirq._compat import cached_property
from cirq_ft import infra


class SelectOracle(infra.GateWithRegisters):
    r"""Abstract base class that defines the API for a SELECT Oracle.

    The action of a SELECT oracle on a selection register $|l\rangle$ and target register
    $|\Psi\rangle$ can be defined as:

    $$
        \mathrm{SELECT} = \sum_{l}|l \rangle \langle l| \otimes U_l
    $$

    In other words, the `SELECT` oracle applies $l$'th unitary $U_{l}$ on the target register
    $|\Psi\rangle$ when the selection register stores integer $l$.

    $$
        \mathrm{SELECT}|l\rangle |\Psi\rangle = |l\rangle U_{l}|\Psi\rangle
    $$
    """

    @property
    @abc.abstractmethod
    def control_registers(self) -> Tuple[infra.Register, ...]:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> Tuple[infra.Register, ...]:
        ...

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )


class PrepareOracle(infra.GateWithRegisters):
    r"""Abstract base class that defines the API for a PREPARE Oracle.

    Given a set of coefficients $\{c_0, c_1, ..., c_{N - 1}\}, the PREPARE oracle is used to encode
    the coefficients as amplitudes of a state $|\Psi\rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle$ using
    a selection register $|i\rangle$. In order to prepare such a state, the PREPARE circuit is also
    allowed to use a junk register that is entangled with selection register.

    Thus, the action of a PREPARE circuit on an input state $|0\rangle$ can be defined as:

    $$
        PREPARE |0\rangle = \sum_{i=0}^{N-1}c_{i}|i\rangle |junk_{i}\rangle
    $$
    """

    @property
    @abc.abstractmethod
    def selection_registers(self) -> Tuple[infra.SelectionRegister, ...]:
        ...

    @cached_property
    def junk_registers(self) -> Tuple[infra.Register, ...]:
        return ()

    @cached_property
    def signature(self) -> infra.Signature:
        return infra.Signature([*self.selection_registers, *self.junk_registers])
