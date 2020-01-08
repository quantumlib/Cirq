from typing import List, Dict, Sequence, Any
from cirq import devices, ops


class PasqalNoiseModel(devices.NoiseModel):
    """A noise model for Pasqal neutral atom device """

    def __init__(self):
        self.noise_op_dict = self.get_default_noise_dict()

    def get_default_noise_dict(self) -> Dict[str, Any]:
        """Returns the current noise parameters"""
        default_noise_dict = {
            'X': ops.depolarize(1e-2),
            'Y': ops.depolarize(1e-2),
            'Z': ops.depolarize(1e-2),
            'CX': ops.depolarize(3e-2),
            'CZ': ops.depolarize(3e-2),
            'CCX': ops.depolarize(8e-2),
            'CCZ': ops.depolarize(8e-2),
        }
        return default_noise_dict

    def noisy_moment(self, moment: ops.Moment,
                     system_qubits: Sequence[ops.Qid]) -> List[ops.Operation]:
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
        for op in moment.operations:
            op_str = get_op_string(op)
            try:
                noise_op = self.noise_op_dict[op_str]
            except KeyError:
                break
            for qubit in op.qubits:
                noise_list.append(noise_op.on(qubit))
        return list(moment) + noise_list







def get_op_string(op_obj):
    """Find the string representation for a given gate
    Args:
        op_obj: Gate object, one of: CNotPowGate,CZPowGate,CCXPowGate,CCZPowGate,YPowGate,ZPowGate,
                                XPowGate,PhasedXPowGate,MeasurementGate,ops.IdentityGate
                                Returns:
                                String representing the gate operations
    """
    if isinstance(op_obj, ops.CNotPowGate) or isinstance(op_obj.gate,ops.CNotPowGate):
        op_str = 'CX'

    elif isinstance(op_obj, ops.CZPowGate) or isinstance(op_obj.gate,ops.CZPowGate):
        op_str = 'CZ'

    elif isinstance(op_obj, ops.CCXPowGate) or isinstance(op_obj.gate,ops.CCXPowGate):
        op_str = 'CCX'

    elif isinstance(op_obj, ops.CCZPowGate) or isinstance(op_obj.gate,ops.CCZPowGate):
        op_str = 'CCZ'

    elif isinstance(op_obj, ops.YPowGate) or isinstance(op_obj.gate,ops.YPowGate):
        op_str = 'Y'

    elif isinstance(op_obj, ops.ZPowGate) or isinstance(op_obj.gate,ops.ZPowGate):
        op_str = 'Z'

    elif isinstance(op_obj, ops.XPowGate) or isinstance(op_obj.gate,ops.XPowGate):
        op_str = 'X'

    elif isinstance(op_obj, ops.IdentityGate) or isinstance(op_obj.gate,ops.IdentityGate):
        op_str = 'I'

    elif isinstance(op_obj, ops.MeasurementGate) or isinstance(op_obj.gate, ops.MeasurementGate):
        op_str = 'Meas'

    else:
        raise ValueError('Got unknown gate:', op_obj)

    return op_str
