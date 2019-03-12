# Copyright 2018 The Cirq Developers
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

"""An optimization pass that removes empty moments from a circuit."""

from cirq import Circuit
from cirq.circuits import circuit as _circuit, Optimizer


class DropEmptyMoments(Optimizer):
    """Removes empty moments from a circuit."""

    def _optimize_circuit(self, circuit: Circuit):
        pass

    def drop_empty_moments(self, drop_empty_moments: bool):
        """
        This is a noop implementation - for this optimization does not make
        sense anything else than drop_empty_moments = True
        """
        pass

    def __call__(self, circuit: _circuit.Circuit):
        self.optimize_circuit(circuit)
