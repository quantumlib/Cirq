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

"""The data structure representing a quantum job.

Job data contains, at minimum, contain a circuit and a parameter sweep of any
parameters contained in the circuit.
"""

from cirq.api.google.v1.params_pb2 import ParameterSweep
from cirq.circuits import Circuit


class Job(object):
    """A circuit coupled with any parameter sweeps and meta-data.

    This class should contain all information needed to submit to Quantum
    Engine.
    """

    def __init__(self, circuit=Circuit(), sweep=ParameterSweep()):
        self.circuit = circuit
        self.sweep = sweep

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.circuit == other.circuit and self.sweep == other.sweep

    def __ne__(self, other):
        return not self == other

    __hash__ = None

    def __repr__(self):
        return "Job(%s,%s)" % (self.circuit, self.sweep)

    def __str__(self):
        return "Job(%s,%s)" % (self.circuit, self.sweep)
