# QC Ware Corp.
# June 2018
#
#

# Imports
import cirq
import numpy as np


def exponentiate_qubit_operator(time, operator, qubits, trotter_steps=0):
    """
    Computes the exponential of an operator O: U = exp(i*time*O)
    If operator does not commute trotterization must be implemented manually
    Currently trotterization is first order

    Args:
        time: integer or float representing the coefficient in the exponent

        operator: operator is of the form dictionary or projectq qubitoperator
                  If provided as a dictionary, the format should be keys corresponding to
                  qubit operators organized as tuples where the first entry 
                  is the qubit acted on and the second a string X, Y, Z for the pauli
                  example:
                  {((0,'Z'), (2,'X')): 3.7, (3,'Y'):2 }

        qubits: qubits should be array of cirq.google.XmonQubit(x, y) type

        trotter_steps: integer for the number of troterization steps
                       to decompose the exponential, e.g.
                       exp(i*t*(A+B)) = Product_n exp(i*t*A/n) exp(i*t*B/n)
                       if trotter_steps is set to 0, it will be set to
                       100*(maximum qubit operator coefficient)

    Returns:
        cirq.circuit() with gates representing the unitary obtained by exponentiating operator argument
    """
    # Check if input is in correct format
    if type(operator) == dict:
        terms_dict = operator.items()
    else:
        raise ValueError('Operator should be a'
                         'a Dict representing the operator and its coefficient')

    # define rotations:
    to_x_basis = cirq.RotYGate(half_turns=.5)
    to_y_basis = cirq.RotXGate(half_turns=-.5)
    from_x_basis = cirq.RotYGate(half_turns=-.5)
    from_y_basis = cirq.RotXGate(half_turns=.5)

    # setup trotter steps
    if trotter_steps == 0:
        trotter_steps = int(100 * max(terms_dict.terms.values()))

    # create circuit
    circuit = cirq.Circuit()

    # trotter loops
    for n in range(trotter_steps):

        # Define Exponentiatiation:
        # loop through operators
        for term, coef in terms_dict:
            moment = []
            basis_change = []
            reverse_basis = []
            cnot_gates = []
            prev_index = None
            highest_target_index = None

            for q_op in term:
                # qubit = q_op[0]
                if q_op[1] == 'X':
                    basis_change.append(to_x_basis.on(qubits[q_op[0]]))
                    reverse_basis.append(from_x_basis.on(qubits[q_op[0]]))

                elif q_op[1] == 'Y':
                    basis_change.append(to_y_basis.on(qubits[q_op[0]]))
                    reverse_basis.append(from_y_basis.on(qubits[q_op[0]]))

                elif q_op[1] == ():
                    continue

                if prev_index is not None:
                    cnot_gates.append(cirq.CNOT.on(qubits[prev_index], qubits[q_op[0]]))

                prev_index = q_op[0]
                highest_target_index = q_op[0]

            moment.append(basis_change)
            moment.append(cnot_gates)
            moment.append(cirq.RotZGate(half_turns=2.0 * (coef / np.pi) * time / trotter_steps).on(qubits[highest_target_index]))
            moment.append(list(reversed(cnot_gates)))
            moment.append(reverse_basis)

            circuit.append(moment)

    # TODO: Implement higher order trotter steps

    return(circuit)
