# Copyright 2019 The Cirq Developers
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

from typing import TYPE_CHECKING, Iterable, Sequence, Union

from cirq import ops, value
from cirq.circuits import circuit

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


class NoiseModel(metaclass=value.ABCMetaImplementAnyOneOf):
    """Replaces operations and moments with noisy counterparts.

    A child class must override *at least one* of the following three methods:

        noisy_moments
        noisy_moment
        noisy_operation

    The methods that are not overridden will be implemented in terms of the ones
    that are.

    Simulators told to use a noise model will use these methods in order to
    dynamically rewrite the program they are simulating.
    """

    def noisy_moments(self, moments: 'Iterable[cirq.Moment]',
                      system_qubits: Sequence['cirq.Qid']
                     ) -> Sequence['cirq.OP_TREE']:
        """Adds possibly stateful noise to a series of moments.

        Args:
            moments: The moments to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            A sequence of OP_TREEs, with the k'th tree corresponding to the
            noisy operations for the k'th moment.
        """
        return list(moments)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']) -> 'cirq.OP_TREE':
        """Adds noise to the operations from a moment.

        Args:
            moment: The moment to add noise to.
            system_qubits: A list of all qubits in the system.

        Returns:
            An OP_TREE corresponding to the noisy operations for the moment.
        """
        return list(moment)

    def noisy_operation(self, operation: 'cirq.Operation') -> 'cirq.OP_TREE':
        """Adds noise to an individual operation.

        Args:
            operation: The operation to make noisy.

        Returns:
            An OP_TREE corresponding to the noisy operations implementing the
            noisy version of the given operation.
        """
        return operation

    def apply_noise(self, input_circuit: 'cirq.Circuit') -> 'cirq.Circuit':
        """Compose a circuit applying defined noise methods to the input.

        If multiple noise methods are defined, they will be inserted in order
        of finest granularity i.e. `noisy_operation` operations before
        `noisy_moment` operations before `noisy_moments` operations.

        TODO
        """
        qubits = list(input_circuit.all_qubits())
        # Append to front of op trees within a queue to avoid in place changes
        op_tree = self.noisy_moments(input_circuit, qubits)
        for i, moment in enumerate(input_circuit):
            op_tree.insert(i, self.noisy_moment(moment, qubits))
            for op in moment:
                op_tree[i].insert(0, self.noisy_operation(op))
        return circuit.Circuit.from_ops(op_tree)


@value.value_equality
class _NoNoiseModel(NoiseModel):
    """A default noise model that adds no noise."""

    def noisy_moments(self, moments: 'Iterable[cirq.Moment]',
                      system_qubits: Sequence['cirq.Qid']):
        return list(moments)

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        return moment

    def noisy_operation(self, operation: 'cirq.Operation'):
        return operation

    def _value_equality_values_(self):
        return None

    def __str__(self):
        return '(no noise)'

    def __repr__(self):
        return 'cirq.NO_NOISE'


class ConstantQubitNoiseModel(NoiseModel):
    """Applies noise to each qubit individually at the end of every moment."""

    def __init__(self, qubit_noise_gate: 'cirq.Gate'):
        if qubit_noise_gate.num_qubits() != 1:
            raise ValueError('noise.num_qubits() != 1')
        self.qubit_noise_gate = qubit_noise_gate

    def noisy_moment(self, moment: 'cirq.Moment',
                     system_qubits: Sequence['cirq.Qid']):
        return list(moment) + [self.qubit_noise_gate(q) for q in system_qubits]


NO_NOISE = _NoNoiseModel()  # type: cirq.NoiseModel
