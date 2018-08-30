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
from typing import Dict, Union, Any

import numpy as np
from cirq.circuits import Circuit
from cirq.ops import RotXGate, RotYGate, RotZGate, CNOT, Pauli, PauliString


def exponentiate_qubit_operator(time: Union[int, float],
                                operator: Dict[PauliString, float],
                                trotter_steps: int = 0):
    """
    Computes the exponential of an operator O: U = exp(i*time*O)
    If operator does not commute trotterization must be implemented manually
    Currently trotterization is first order

    Args:
        time: integer or float representing the coefficient in the exponent

        operator: Operator input as a dictionary where keys are
                    cirq.ops.PauliString type and values are coefficients
                  example:
                  paulistring = cirq.PauliString(qubit_pauli_map=
                    {qubits[0]:cirq.Pauli.X, qubits[1]: (cirq.Pauli.Z)})
                  paulistring2 = cirq.PauliString(qubit_pauli_map=
                    {qubits[0]:cirq.Pauli.Z, qubits[2]: (cirq.Pauli.Y)})
                  operator = {paulistring: 1.234, paulistring2: -5.678}

        trotter_steps: integer for the number of trotterization steps
                       to decompose the exponential, e.g.
                       exp(i*t*(A+B)) = Product_n exp(i*t*A/n) exp(i*t*B/n)
                       if trotter_steps is set to 0, it will be set to
                       100*(maximum qubit operator coefficient)

    Returns:
        cirq.circuit() with gates representing the unitary
        obtained by exponentiating operator argument
    """

    # Check if input is in correct format
    if isinstance(operator, dict):
        terms_dict = operator.items()
    else:
        raise ValueError('Operator should be a'
                         'Dict representing with PauliString as key'
                         'and coefficient of term as dict.values')

    # define rotations:
    basis_rotation = {
        Pauli.X: RotYGate(half_turns=.5),
        Pauli.Y: RotXGate(half_turns=-.5),
        Pauli.Z: None,
        (): None
    }

    # rotations back to original basis
    undo_rotation = {
        Pauli.X: RotYGate(half_turns=-.5),
        Pauli.Y: RotXGate(half_turns=.5)
    }

    # setup trotter steps
    if trotter_steps == 0:
        trotter_steps = int(100 * np.max(np.absolute(list(
            operator.values()))))

    # create circuit
    circuit = Circuit()

    # trotter loops
    for _ in range(trotter_steps):

        # Define Exponentiatiation:
        # loop through operators
        for term, coef in terms_dict:
            moment = []
            basis_change = []
            reverse_basis = []
            cnot_gates: List[Any] = []
            prev_qubit = None
            highest_target_qubit = None

            # skip identity
            # if term == ():
            #    continue

            for qubit, pauli in term.items():
                rotation = basis_rotation[pauli]
                if rotation is not None:
                    change = rotation.on(qubit)
                    undo_change = undo_rotation[pauli].on(qubit)
                    basis_change.append(change)
                    reverse_basis.append(undo_change)

                if prev_qubit is not None:
                    cnot_gates.append(CNOT.on(prev_qubit,
                                              qubit))

                prev_qubit = qubit
                highest_target_qubit = qubit

            moment.append(basis_change)
            moment.append(cnot_gates)
            moment.append(RotZGate(half_turns=2.0 * (coef / np.pi)
                                   * time / trotter_steps).on(
                highest_target_qubit))
            moment.append(list(reversed(cnot_gates)))
            moment.append(reverse_basis)

            circuit.append(moment)

    # TODO: Implement higher order trotter steps

    return circuit
