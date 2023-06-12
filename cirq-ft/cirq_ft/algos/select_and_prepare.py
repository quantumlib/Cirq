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
from functools import cached_property

from cirq_ft import infra


class SelectOracle(infra.GateWithRegisters):
    @property
    @abc.abstractmethod
    def control_registers(self) -> infra.Registers:
        ...

    @property
    @abc.abstractmethod
    def selection_registers(self) -> infra.SelectionRegisters:
        ...

    @property
    @abc.abstractmethod
    def target_registers(self) -> infra.Registers:
        ...

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers(
            [*self.control_registers, *self.selection_registers, *self.target_registers]
        )


class PrepareOracle(infra.GateWithRegisters):
    @property
    @abc.abstractmethod
    def selection_registers(self) -> infra.SelectionRegisters:
        ...

    @cached_property
    def junk_registers(self) -> infra.Registers:
        return infra.Registers([])

    @cached_property
    def registers(self) -> infra.Registers:
        return infra.Registers([*self.selection_registers, *self.junk_registers])
