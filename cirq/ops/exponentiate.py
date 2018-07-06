# Copyright 2018 The Cirq Developers
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

""" Exponentiation tool for Pauli Operators""" 

# Imports
from typing import Optional, Dict, Tuple, List, Union

import numpy as np
from cirq import Circuit
from cirq.ops import RotXGate, RotYGate, RotZGate, CNOT
from cirq.google import XmonQubit

def exponentiate_qubit_operator(time: Union[int,float], 
                                operator: Dict[Tuple[int,str], float],
                                qubits: List[XmonQubit], 
                                trotter_steps: int = 0):
    """
    Computes the exponential of an operator O: U = exp(i*time*O)
    If operator does not commute trotterization must be implemented manually
    Currently trotterization is first order

    Args:
        time: integer or float representing the coefficient in the exponent

        operator: operator is of the form dictionary or projectq qubitoperator
                  If provided as a dictionary, the format should be keys 
                  corresponding to qubit operators organized as tuples where
                  the first entry is the qubit acted on and the second a 
                  string X, Y, Z for the pauli
                  example:
                  {((0,'Z'), (2,'X')): 3.7, ((3,'Y'),):2 }

        qubits: qubits should be array of cirq.google.XmonQubit(x, y) type

        trotter_steps: integer for the number of troterization steps
                       to decompose the exponential, e.g.
                       exp(i*t*(A+B)) = Product_n exp(i*t*A/n) exp(i*t*B/n)
                       if trotter_steps is set to 0, it will be set to
                       100*(maximum qubit operator coefficient)

    Returns:
        cirq.circuit() with gates representing the unitary 
        obtained by exponentiating operator argument
    """
    # Check if input is in correct format
    if type(operator) == dict:
        terms_dict = operator.items()
    else:
        raise ValueError('Operator should be a'
                         'Dict representing the operator and its coefficient')

    # define rotations:
    basis_rotation = {
        'X' : RotYGate(half_turns=.5),
        'Y' : RotXGate(half_turns=-.5),
        'Z' : None,
        () : None
        }

    # rotations back to original basis
    undo_rotation = {
        'X' : RotYGate(half_turns=-.5),
        'Y' : RotXGate(half_turns=.5)
        }


#        from_x_basis = cirq.RotYGate(half_turns=-.5)
#        from_y_basis = cirq.RotXGate(half_turns=.5)

    # setup trotter steps
    if trotter_steps == 0:
        trotter_steps = abs(int(100 * max(dict(terms_dict).values())))

    # create circuit
    circuit = Circuit()

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
            
            # skip identity
            if term == ():
                continue
            
            # if format for single paulis is wrong, correct it:
            if not isinstance(term[0],tuple):
                term = (term,)

            for qubit_number, op in term:
                rotation = basis_rotation[op]
                if rotation is not None:
                    change = rotation.on(qubits[qubit_number])
                    undo_change = undo_rotation[op].on(qubits[qubit_number])
                    basis_change.append(change)
                    reverse_basis.append(undo_change)


#                # qubit = q_op[0]
#                if q_op[1] == 'X':
#                    basis_change.append(to_x_basis.on(qubits[q_op[0]]))
#                    reverse_basis.append(from_x_basis.on(qubits[q_op[0]]))
#
#                elif q_op[1] == 'Y':
#                    basis_change.append(to_y_basis.on(qubits[q_op[0]]))
#                    reverse_basis.append(from_y_basis.on(qubits[q_op[0]]))
#
#                elif q_op[1] == ():
#                    continue

                if prev_index is not None:
                    cnot_gates.append(CNOT.on(qubits[prev_index],
                                                   qubits[qubit_number]))

                prev_index = qubit_number
                highest_target_index = qubit_number

            moment.append(basis_change)
            moment.append(cnot_gates)
            moment.append(RotZGate(half_turns=2.0 * (coef / np.pi) * 
                                        time / trotter_steps).on(
                                        qubits[highest_target_index]))
            moment.append(list(reversed(cnot_gates)))
            moment.append(reverse_basis)

            circuit.append(moment)

    # TODO: Implement higher order trotter steps

    return circuit
