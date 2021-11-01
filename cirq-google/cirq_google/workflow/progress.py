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
import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq_google as cg


class _WorkflowLogger(abc.ABC):
    def initialize(self):
        """Run at the start of an execution loop."""

    def consume_one(self, exe_result: 'cg.ExecutableResult',
                    shared_rt_info: 'cg.SharedRuntimeInfo'):
        """Run when a result is done."""

    def finalize(self):
        """Run at the end"""


class _PrintLogger(_WorkflowLogger):
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.i = 0

    def initialize(self):
        print()

    def consume_one(self, exe_result: 'cg.ExecutableResult',
                    shared_rt_info: 'cg.SharedRuntimeInfo'):
        print(f'\r{self.i + 1} / {self.n_total}', end='', flush=True)
        self.i += 1

    def finalize(self):
        print()
