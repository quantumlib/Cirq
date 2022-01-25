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

"""Progress and logging facilities for the quantum runtime."""

import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cirq_google as cg


class _WorkflowLogger(abc.ABC):
    """Implementers of this class can provide logging and progress information
    for execution loops."""

    def initialize(self):
        """Initialization logic at the start of an execution loop."""

    def consume_result(
        self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Consume executable results as they are completed.

        Args:
            exe_result: The completed `cg.ExecutableResult`.
            shared_rt_info: A reference to the `cg.SharedRuntimeInfo` for this
                execution at this point.
        """

    def finalize(self):
        """Finalization logic at the end of an execution loop."""


class _PrintLogger(_WorkflowLogger):
    def __init__(self, n_total: int):
        self.n_total = n_total
        self.i = 0

    def initialize(self):
        """Write a newline at the start of an execution loop."""
        print()

    def consume_result(
        self, exe_result: 'cg.ExecutableResult', shared_rt_info: 'cg.SharedRuntimeInfo'
    ):
        """Print a simple count of completed executables."""
        print(f'\r{self.i + 1} / {self.n_total}', end='', flush=True)
        self.i += 1

    def finalize(self):
        """Write a newline at the end of an execution loop."""
        print()
