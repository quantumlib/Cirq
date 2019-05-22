from cirq import LineQubit, IonDevice, Duration
from cirq import depolarize
"""Current device parameters for the AQT/UIBK ion trap device

The device is based on a linear calcium ion string with
arbitrary connectivity. For more information see:

https://quantumoptics.at/en/publications/journal-articles.html

https://iopscience.iop.org/article/10.1088/1367-2630/15/12/123012/meta

The native gate set is local gates: X,Y, and XX entangling gates

"""


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
default_noise_dict['crosstalk'] = 0.03  # type: ignore
