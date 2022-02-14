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

"""Pushes 180 degree rotations around axes in the XY plane later in the circuit.
"""

from cirq import _compat, circuits, transformers


@_compat.deprecated_class(deadline='v1.0', fix='Use cirq.eject_phased_paulis instead.')
class EjectPhasedPaulis:
    """Pushes X, Y, and PhasedX gates towards the end of the circuit.

    As the gates get pushed, they may absorb Z gates, cancel against other
    X, Y, or PhasedX gates with exponent=1, get merged into measurements (as
    output bit flips), and cause phase kickback operations across CZs (which can
    then be removed by the EjectZ optimization).
    """

    def __init__(self, tolerance: float = 1e-8, eject_parameterized: bool = False) -> None:
        """Inits EjectPhasedPaulis.

        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations gates with a
                 threshold determined by this tolerance.
            eject_parameterized: If True, the optimization will attempt to eject
                parameterized gates as well.  This may result in other gates
                parameterized by symbolic expressions.
        """
        self.tolerance = tolerance
        self.eject_parameterized = eject_parameterized

    def optimize_circuit(self, circuit: circuits.Circuit):
        circuit._moments = [
            *transformers.eject_phased_paulis(
                circuit, atol=self.tolerance, eject_parameterized=self.eject_parameterized
            )
        ]
