from typing import List, Dict, Sequence, Any

import cirq

from . import PasqalDevice

class PasqalNoiseModel(cirq.devices.NoiseModel):
    """A noise model for Pasqal neutral atom device """

    def __init__(self):
        self.noise_op_dict = self.get_default_noise_dict()

    def get_default_noise_dict(self) -> Dict[str, Any]:
        """Returns the current noise parameters"""
        default_noise_dict = {
            str(cirq.ops.YPowGate): cirq.ops.depolarize(1e-2),
            str(cirq.ops.ZPowGate): cirq.ops.depolarize(1e-2),
            str(cirq.ops.XPowGate): cirq.ops.depolarize(1e-2),
            str(cirq.ops.PhasedXPowGate): cirq.ops.depolarize(1e-2),
            str(cirq.ops.CNotPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
            str(cirq.ops.CZPowGate(exponent=1)): cirq.ops.depolarize(3e-2),
            str(cirq.ops.CCXPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
            str(cirq.ops.CCZPowGate(exponent=1)): cirq.ops.depolarize(8e-2),
        }
        return default_noise_dict

    def noisy_moment(self, moment: cirq.ops.Moment,
                     system_qubits: Sequence[cirq.ops.Qid]) -> List[cirq.ops.Operation]:
        """Returns a list of noisy moments.
        The model includes
        - Depolarizing noise with gate-dependent strength
        Args:
            moment: ideal moment
            system_qubits: List of qubits
        Returns:
            List of ideal and noisy moments
        """
        noise_list = []
        for op in moment:
            op_str = get_op_string(op)
            try:
                noise_op = self.noise_op_dict[op_str]
            except KeyError:
                noise_op = cirq.ops.depolarize(5e-2)
            for qubit in op.qubits:
                noise_list.append(noise_op.on(qubit))
        return list(moment) + noise_list


def get_op_string(cirq_op):
    if not isinstance(cirq_op, cirq.ops.Operation):
        raise ValueError('Got unknown operation:', cirq_op)

    if not PasqalDevice.is_pasqal_device_op(cirq_op) \
            or isinstance(cirq_op.gate, cirq.ops.MeasurementGate):
        # TODO: Is there noise for measurements?
        raise ValueError('Got unknown operation:', cirq_op)

    return str(cirq_op.gate)
