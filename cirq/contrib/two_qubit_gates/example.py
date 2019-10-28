from matplotlib import pyplot as plt
import numpy as np

from cirq import FSimGate, unitary
from cirq.contrib.two_qubit_gates.gate_compilation import gate_product_tabulation
from cirq.contrib.two_qubit_gates.math_utils import random_two_qubit_unitaries_and_kak_vecs, unitary_entanglement_fidelity

base = unitary(FSimGate(np.pi / 4, np.pi / 24))

max_infidelity = 5e-3
tabulation = gate_product_tabulation(base, max_infidelity)

unitaries, _ = random_two_qubit_unitaries_and_kak_vecs(1000)
target = unitaries[0]

infidelities = []
failed_infidelities = []
for target in unitaries:
    local_us, actual, success = tabulation.compile_two_qubit_gate(target)
    infidelity = 1 - unitary_entanglement_fidelity(target, actual)
    if success:
        infidelities.append(infidelity)
    else:
        failed_infidelities.append(infidelity)

infidelities = np.array(infidelities)
failed_infidelities = np.array(failed_infidelities)

plt.figure()
plt.hist(infidelities, bins=25, range=[0, max_infidelity * 1.1])
ylim = plt.ylim()
plt.plot([max_infidelity] * 2, ylim, '--',
         label='Maximum tabulation infidelity')
plt.xlabel('Compiled gate infidelity vs target')
plt.ylabel('Counts')
plt.legend()
