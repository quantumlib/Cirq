# script to run an the implementation of QAOA

from qaoa_function import qaoa_solver
from cirq.ops import PauliString, Pauli
from cirq.line import LineQubit


if __name__ == '__main__':

    # set number of qubits used in routine
    n_qubits = 3

    # set the objective function to be optimized.
    # keys should be operator, values should be the coefficient
    # for the identity operator insert () as key
    qubits = [LineQubit(i) for i in range(n_qubits)]
    objective_function = {PauliString(
        qubit_pauli_map={qubits[0]:Pauli.Z,
            qubits[1]:Pauli.Z}): -1/2,
                          (): 1/2}

    # set optimization method
    optimization_method = 'nelder-mead'

    # set value of p (number of alternating applications of
    # Unitaries constructed from the objective function and mixing op
    p = 1

    # set mixing operator, optional
    # mixing = {PauliString(...):...}
    mixing = 'X'

    # set expectation method (input integer for sampling)
    expectation_method = 'wavefunction'

    solution = qaoa_solver(n_qubits=n_qubits,
                objective_function=objective_function,
                optimization_method=optimization_method,
                p=p,
                mixing=mixing, expectation_method=expectation_method)

    print(solution)
