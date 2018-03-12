# Copyright 2017 Google LLC
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
import random
from cirq.api.google.v1.params_pb2 import ParameterSweep
from cirq.api.google.v1.params_pb2 import SingleParameterSweep
from cirq.circuits.circuit import Circuit
from cirq.circuits.circuit import Moment
from cirq.contrib.jobs import Job
from cirq.google import xmon_gates
from cirq.study.parameterized_value import ParameterizedValue


class DepolarizerChannel(object):
    """Depolarizing Channel Transformer for error simulation.

    This class simulates errors in a quantum circuit by randomly
    introducing error gates in between each moment.  This simulates
    noise by decohering random qubits at each step.  This transform
    is intended only for classical simulations only.

    This class currently only supports adding a Pauli-Z gate at
    each step.

    If the job already contains a parameter sweep, this will create
    the error sweep as a cartesian product (i.e. a "factor") of the
    existing parameter sweep.  For example, if the job contains a
    parameter with 4 values, the new job will run through the same
    error scenarios for each of the 4 values.

    Attributes:
      probability: Probability of a qubit being affected in a given moment
      repetitions: Number of simulations to create.
    """

    # Prefix for ParameterizedValues related to error simulation
    _parameter_name = 'error_parameter'

    def __init__(self, probability=0.001, repetitions=1):
        self.p = probability
        self.repetitions = repetitions

    def transform_job(self, job):
        """Creates a new job object with depolarizing channel.

        This job will contain the existing Job's circuit with an error gate per
        qubit at every moment.  Creates the parameter sweep for each gate and
        populates with random values as per the specifications of the
        depolarizer channel.

        Does not yet support augmenting jobs that already have parameter sweeps.

        Args:
            job: Job object to transform

        Returns:
            A new Job object that contains a circuit with up to double the
            moments as the original job, with every other moment being a
            moment containing error gates.  It will also contain a
            ParameterSweep containing values for each error gate.  Note that
            moments that contain no error gates for any repetition will be
            automatically omitted.
        """
        # A set for quick lookup of pre-existing qubits
        qubit_set = set()
        # A list with deterministic qubit order
        qubit_list = []
        circuit = job.circuit

        # Retrieve the set of qubits used in the circuit
        for moment in circuit.moments:
            for op in moment.operations:
                for qubit in op.qubits:
                  if qubit not in qubit_set:
                    qubit_set.add(qubit)
                    qubit_list.append(qubit)

        # Add error circuits
        moments = []
        parameter_number = 0
        sweep = ParameterSweep()
        sweep.CopyFrom(job.sweep)
        if sweep.repetitions <= 0:
            # If unset, set to 1 repetition.
            sweep.repetitions = 1
        error_factor = len(sweep.sweep.factors)
        sweep.sweep.factors.add()
        add_gate = False
        for moment in circuit.moments:
            moments.append(moment)

            points = {}
            gate_needed = {}
            for q in qubit_list:
                points[q] = []
                for _ in range(self.repetitions):
                    if random.random() < self.p:
                        add_gate = True
                        gate_needed[q] = True
                        points[q].append(1.0)
                    else:
                        points[q].append(0.0)

            if add_gate:
                moments.append(self._error_moment(parameter_number,
                                                  qubit_list, gate_needed))

                for q in qubit_list:
                    if gate_needed[q]:
                        sps = self._single_parameter_sweep(parameter_number,
                                                           points[q])
                        sweep.sweep.factors[error_factor].sweeps.extend([sps])
                    parameter_number += 1

        if add_gate:
            new_job = Job(Circuit(moments), sweep)
        else:
            new_job = Job(Circuit(moments))

        return new_job

    def _error_moment(self, current_parameter_number, qubit_list, gate_needed):
        """Creates a moment of error gates.

        Args:
            current_parameter_number: The current number of error parameters
                already existing.  Used for naming purposes.
            qubit_list: List of qubits to add error gates for.
            gate_needed: dictionary of qubits.  Value is true if an error
                gate should be added for the qubit.
        Returns:
            a moment that includes parameterized errors gates.  The
            parameters will be named with monotonically increasing error
            parameterized values.
        """
        error_gates = []
        for q in qubit_list:
            if gate_needed[q]:
                new_key = self._parameter_name + str(current_parameter_number)
                current_parameter_number += 1
                new_parameter = ParameterizedValue(key=new_key)
                error_gates.append(xmon_gates.ExpZGate(
                    half_turns=new_parameter).on(q))
        return Moment(error_gates)

    def _single_parameter_sweep(self, current_parameter_number, point_list):
        """Creates a single parameter sweep.

        Args:
           current_parameter_number: The current number of error parameters
                already existing.  Used for naming parameterized values.
           point_list: list of floats to convert to SweepPoints
        Returns:
          a SingleParameterSweep for a given qubit based on randomized results.
        """
        key = self._parameter_name + str(current_parameter_number)
        sps = SingleParameterSweep()
        sps.parameter_name = key
        for point in point_list:
             sps.sweep_points.points.append(point)
        return sps
