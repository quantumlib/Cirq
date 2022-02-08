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

"""An optimization pass that pushes Z gates later and later in the circuit."""

from cirq import circuits, transformers

# from cirq.transformers import eject_z
from cirq._compat import deprecated_class


@deprecated_class(deadline='v1.0', fix='Use cirq.eject_z instead.')
class EjectZ:
    """Pushes Z gates towards the end of the circuit.

    As the Z gates get pushed they may absorb other Z gates, get absorbed into
    measurements, cross CZ gates, cross W gates (by phasing them), etc.
    """

    def __init__(self, tolerance: float = 0.0, eject_parameterized: bool = False) -> None:
        """Inits EjectZ.

        Args:
            tolerance: Maximum absolute error tolerance. The optimization is
                 permitted to simply drop negligible combinations of Z gates,
                 with a threshold determined by this tolerance.
            eject_parameterized: If True, the optimization will attempt to eject
                parameterized Z gates as well.  This may result in other gates
                parameterized by symbolic expressions.
        """
        self.tolerance = tolerance
        self.eject_parameterized = eject_parameterized

    def optimize_circuit(self, circuit: circuits.Circuit):
        circuit._moments = [
            *transformers.eject_z(
                circuit, atol=self.tolerance, eject_parameterized=self.eject_parameterized
            )
        ]
