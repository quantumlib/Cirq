from typing import Tuple, Union, List

import cirq


class TwoQubitNoiseModel(cirq.NoiseModel):
    """
    The two qubit noise channel applies a different noise channel to the circuit
    dependent on whether the gate applied is applied to one or two qubits.
    Can be used to apply the same symmetric depolarising channel with different depolarising probabilities,
    or a different type of channel altogether, dependent on the gate applied.
    """

    def __init__(self, single_qubit_noise_gate: cirq.Gate,
                 two_qubit_noise_gate: cirq.Gate):
        if single_qubit_noise_gate.num_qubits() != 1:
            raise ValueError(
                'The noise gate provided to single_qubit_noise_gate has number of qubits != 1.'
            )
        if two_qubit_noise_gate.num_qubits() != 2:
            raise ValueError(
                'The noise gate provided to two_qubit_noise_gate has number of qubits != 2.'
            )
        self.single_qubit_noise_gate = single_qubit_noise_gate
        self.two_qubit_noise_gate = two_qubit_noise_gate

    def noisy_operation(
            self, operation: cirq.Operation
    ) -> Tuple[cirq.Operation, Union[List[cirq.GateOperation], cirq.
                                     GateOperation]]:
        """
        Checks if the gate in the operation is a one- or two- qubit gate,
        and applies self.single_qubit_noise_gate or self.two_qubit_noise_gate appropriately.
        If the operation has > 2 qubits, applies the single qubit noise gate to all.
        :param operation: The operation to apply noise to
        :return: The supplied operation and the noise gate(s) to apply.
        """
        n_qubits = len(operation.qubits)
        if n_qubits == 1:
            return operation, self.single_qubit_noise_gate(operation.qubits[0])
        elif n_qubits == 2:
            return operation, self.two_qubit_noise_gate(operation.qubits[0],
                                                        operation.qubits[1])
        else:
            return operation, [
                self.single_qubit_noise_gate(q) for q in operation.qubits
            ]
