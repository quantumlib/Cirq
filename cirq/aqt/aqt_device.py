from cirq import LineQubit, IonDevice, Duration
from cirq import depolarize
"""Current device parameters for the AQt ion trap device"""


def get_aqt_device(no_qubits: int):
    """Returns an AQT ion device
    :Args:
        no_qubits: number of qubits
    Returns:
         IonDevice, qubit_list
    """
    qubit_list = LineQubit.range(no_qubits)
    us = 1000 * Duration(nanos=1)
    ion_device = IonDevice(measurement_duration=100 * us,
                           twoq_gates_duration=200 * us,
                           oneq_gates_duration=10 * us,
                           qubits=qubit_list)
    return ion_device, qubit_list


default_noise_dict = {}
default_noise_dict['X'] = depolarize(1e-3)
default_noise_dict['Y'] = depolarize(1e-3)
default_noise_dict['MS'] = depolarize(1e-2)
default_noise_dict['crosstalk'] = 0.03
