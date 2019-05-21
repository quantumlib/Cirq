from cirq import LineQubit, IonDevice,Duration

"""Current device parameters for the AQt ion trap device"""

def get_aqt_device(no_qubits: int) -> [IonDevice, list]:
    """Returns an AQT ion device
    :param no_qubits: number of qubits
    :return: IonDevice, qubit_list
    """
    qubit_list = LineQubit.range(no_qubits)
    us = 1000*Duration(nanos=1)
    ion_device = IonDevice(measurement_duration=100*us, twoq_gates_duration=200*us,
                        oneq_gates_duration=10*us, qubits=qubit_list)
    return ion_device, qubit_list
