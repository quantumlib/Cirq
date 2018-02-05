# Copyright 2018 Google LLC
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

"""Defines the OptimizationPass type."""

import abc
from typing import Optional

from cirq import ops
from cirq.circuits import Circuit


class OptimizationPass:
    """Rewrites a circuit's operations to make them better."""

    @abc.abstractmethod
    def optimize_circuit(self, circuit: Circuit):
        """Rewrites the given circuit to make it better.

        Args:
            circuit: The circuit to improve.
        """
        pass


class PointOptimizer:
    """Makes circuit improvements focused on a specific location."""

    @abc.abstractmethod
    def optimize_at(self, circuit: Circuit, index: int,
                    op: ops.Operation) -> Optional[int]:
        """Rewrites the given circuit to improve it, focusing on one spot.

        Args:
            circuit: The circuit to improve.
            index: The index of the moment with the operation to focus on.
            op: The operation to focus improvements upon.

        Returns:
            The new index of the moment that was being optimized, or None if it
            can be assumed to be in the same place.
        """
        pass

    def optimize_circuit(self, circuit: Circuit):
        i = 0
        while i < len(circuit.moments):
            for op in circuit.moments[i].operations:
                # Note: Circuit may have been mutated.
                if (i < len(circuit.moments) and
                        op in circuit.moments[i].operations):
                    r = self.optimize_at(circuit, i, op)
                    if r is not None:
                        i = max(r, i)
            i += 1
