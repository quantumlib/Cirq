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

"""Error simulator that adds randomly activated error gates after every moment.
"""

import numpy as np

from cirq.circuits.circuit import Circuit
from cirq.circuits.circuit import Moment
from cirq.contrib.jobs import Job
from cirq.study.sweeps import Points, Zip
from cirq.value import Symbol
from cirq import ops


class DepolarizerChannel(object):
    """Depolarizing Channel Transformer for error simulation.

    This class simulates errors in a quantum circuit by randomly
    introducing error gates in between each moment.  This simulates
    noise by decohering random qubits at each step.  This transform
    is intended only for classical simulations only.

    This class currently defaults to adding a Pauli-Z gate at
    each step but the gate set to add is configurable.  Note that
    all gates must be XmonGates or have a half_turns parameter
    that will be overwritten by this class with a parameterized
    value.

    If the job already contains a parameter sweep, this will create
    the error sweep as a cartesian product (i.e. a "factor") of the
    existing parameter sweep.  For example, if the job contains a
    parameter with 4 values, the new job will run through the same
    error scenarios for each of the 4 values.

    Attributes:
        probability: Probability of a qubit being affected in a given moment
        realizations: Number of simulations to create.
        depolarizing_gates: A list of XmonGates to include at every moment.
    """

    # Prefix for symbols related to error simulation
    _parameter_name = 'error_parameter'

    def __init__(self, probability=0.001, realizations=1,
                 depolarizing_gates=(ops.Z,)):
        self.p = probability
        self.realizations = realizations
        self.depolarizing_gates = depolarizing_gates

    def transform_job(self, job):
        """Creates a new job object with depolarizing channel.

        This job will contain the existing Job's circuit with an error gate per
        qubit at every moment.  Creates the parameter sweep for each gate and
        populates with random values as per the specifications of the
        depolarizer channel.

        Args:
            job: Job object to transform

        Returns:
            A new Job object that contains a circuit with up to double the
            moments as the original job, with every other moment being a
            moment containing error gates.  It will also contain a Sweep
            containing values for each error gate.  Note that moments that
            contain no error gates for any repetition will be automatically
            omitted.
        """
        # A set for quick lookup of pre-existing qubits
        qubit_set = set()
        # A list with deterministic qubit order
        qubit_list = []
        circuit = job.circuit

        # Retrieve the set of qubits used in the circuit
        for moment in circuit:
            for op in moment.operations:
                for qubit in op.qubits:
                    if qubit not in qubit_set:
                        qubit_set.add(qubit)
                        qubit_list.append(qubit)

        # Add error circuits
        moments = []
        error_number = 0
        error_sweep = Zip()

        for moment in circuit:
            moments.append(moment)

            for gate in self.depolarizing_gates:
                error_gates = []
                for q in qubit_list:
                    errors = np.random.random(self.realizations) < self.p
                    if any(errors):
                        key = self._parameter_name + str(error_number)
                        new_error_gate = gate**Symbol(key)
                        error_gates.append(new_error_gate.on(q))
                        error_sweep += Points(key, list(errors * 1.0))
                        error_number += 1

                if error_gates:
                    moments.append(Moment(error_gates))

        sweep = job.sweep
        if error_sweep:
            sweep *= error_sweep

        return Job(Circuit(moments), sweep, job.repetitions)
