# QC Ware Corp.
# June 2018
#
#

# imports
import cirq
import numpy as np
from cirq.google import XmonSimulator, XmonMeasurementGate
from cirq import Circuit
from cirq import RotXGate, RotYGate, MeasurementGate

def expectation_value(circuit, operator, method='wavefunction',
                      n_samples=None, measurement=False, repetitions=100, quadratic_z=False):
    """
    Calculates the expactation value of cost function (operator)

    Args:
        circuit: circ.Circuit type which prepares the state we are interested
                 in calculating the expectation value of

        operator: Operator input as a dictionary where keys are tuples corresponding
                  to the pauli matrices and qubits and the values are the coefficients
                  example: 
                  operator = {((0,'X'), (3,'Z')): 5, (2,'Y'):1.234
                  we are interested in calculating the expectation value of

        method: sampling or wavefunction
                wavefunction accesses the final state configuration and
                computes expectation from probabilities

                sampling is for future incorporation with QPU (not simulator)

        n_samples: number of runs for sampling

        measurement: if set to True, then measurement gates are applied to simulator
                     and expectation is obtained by sampling

        repetitions: number of times circuit is run on simulator

        quadratic_z: Boolean that determines whether operator only has terms quadratic in Pauli Z
                     If true, computation is faster

    Returns:
        expectation value of operator argument

    """

    qubits = list(circuit.qubits())
    sim = cirq.google.XmonSimulator()

    # Helper function to remove measrument gates
    # ensures there are no measurement gates since rotation gates must still be added:
    def no_meas(circuit):
        cir2 = cirq.Circuit()
        for moment in circuit.moments:
            new_moment = []
            for op in moment.operations:
                if not isinstance(op.gate, cirq.MeasurementGate):
                    new_moment.append(op)

            cir2.append(new_moment)

        return(cir2)

    # If measurement is true, effective sampling is done on the simulator
    # note that the number of samples is the 'repetitions' variable
    if measurement is True and quadratic_z is False:

        circuit = no_meas(circuit)

        expectation = 0
        for term, coef in operator.items():

            # check if identity:
            identity_coeficient = 0
            if term == ():
                identity_coeficient = coef
                continue

            # Add appropriate rotation gates to circuit
            # and add measurement gates
            q_dict = dict()  # dictionary of qubits to be measured and its indices

            for pauli in term:

                if pauli[1] == 'X':
                    circuit.append(cirq.RotYGate(half_turns=-1 / 2).on(qubits[pauli[0]]))

                elif pauli[1] == 'Y':
                    circuit.append(cirq.RotXGate(half_turns=+1 / 2).on(qubits[pauli[0]]))

                # After rotation we can re-add measurement gates
                q_dict['{}'.format(pauli[0])] = qubits[pauli[0]]

            ##############
            # Function to add measurement gates to qubits
            def measurement_gates(circuit, q_dict):

                meas = [XmonMeasurementGate(key='q{}'.format(i)).on(
                        qubit) for i, qubit in q_dict.items()]

                circuit.append(meas)

                return(circuit)
            ###############

            # Add Measurement gates:

            circuit = measurement_gates(circuit, q_dict)
            print(circuit)
            # run circuit:
            results = sim.run(circuit, repetitions=repetitions)

            operator_meas = 1

            for qubit_key in results.measurements.keys():
                operator_meas *= results.measurements[qubit_key][:, 0].astype(int) * (-2) + 1

            mean_measurement = np.mean(operator_meas)

            expectation += mean_measurement * coef

        # add identity term:
        expectation += identity_coeficient

        return(expectation)

    # default 100 samples
    # Note samples is different than repetitions
    # 'sampling' is for not useful yet
    if method == 'sampling' and n_samples is None:
        n_samples = 100

    # If measurement is true, effective sampling is done on the simulator
    # note that the number of samples is the 'repetitions' variable
    if measurement is True and quadratic_z is True:

        # Check whether circuit already has measurement gates:
        # only applicable if calculating expecation by measurement

        last_moment_index = len(circuit.moments - 1)
        last_operations = circuit.moments[last_moment_index].operations

        measurement_qubits = list(qubits)
        indices_measurement = list(range(len(qubits)))

        for operation in last_operations:
            if isinstance(operation.gate, cirq.MeasurementGate):
                measurement_qubits.remove(operation.qubits[0])
                indices_measurement.remove(qubits.index(operation.qubits[0]))

        assert len(indices_measurement) == len(measurement_qubits), 'Indices do not match qubits'

        q_dict = {}
        for i in indices_measurement:
            q_dict['{}'.format(i)] = measurement_qubits[i]

        # Function to add measurement gates to qubits
        def measurement_gates(circuit, q_dict):
            # note q_dict is a dictionary with indices and qubits

            meas = [XmonMeasurementGate(key='q{}'.format(i)).on(
                    qubit) for i, qubit in q_dict.items()]

            circuit.append(meas)

            return(circuit)


        # add measurement gate to appropriate qubits

        circuit = measurement_gates(circuit, q_dict)

        # run circuit and store results
        results = sim.run(circuit, repetitions=repetitions)
        results_dict = results.measurements

        expectation_samples = []
        for rep in range(repetitions):
            rep_term = rep
            expectation = 0

            # loop over terms in cost function
            for term, coef in operator.items():
                # skip over identity terms
                if len(term) == 0:
                    print('Ignoring overall constant to cost')
                    continue

                # check that terms only contain Z, as expected in maxcut
                assert term[0][1] == 'Z', 'Only hamiltonian for max cut supported, ops must be Z'
                if len(term) == 2:
                    assert term[1][1] == 'Z', 'Only hamiltonian for max cut supported, ops must be Z'

                # caculates expectation

                q1 = term[0][0]
                if len(term) == 2:
                    q2 = term[1][0]
                else:
                    q2 = q1

                x1 = int(results_dict['q{}'.format(q1)][rep_term][0])
                x2 = int(results_dict['q{}'.format(q2)][rep_term][0])

                expectation += x1 * coef * x2

            expectation_samples.append(expectation)

        expectation_mean = np.mean(expectation_samples)

        # If we want to add an option to see what the expectation value is
        # during long qaoa runs:

        #if print_runs:
        #    print('current cost is = ', expectation_mean)

        return(expectation_mean)

    if method == 'wavefunction':

        # number of qubits
        n_qubits = len(qubits)

        # setup the bit strings for each final state:
        bits = [bin(i)[2:].zfill(n_qubits) for i in range(2**n_qubits)]

        # ensures no measurements are performed:
        circuit = no_meas(circuit)

        # runs circuit:
        results = sim.simulate(circuit)

        # setup the probabilities for each state
        final_state = results.final_state

        expectation = 0

        # Arrays to help in computation

        pauli_x = np.array([[0, 1], [1, 0]])
        pauli_y = np.array([[0, -1j], [1j, 0]])
        pauli_z = np.array([[1, 0], [0, -1]])
        identity = np.identity(2)

        for term, coef in operator.items():
            # reset operator
            full_op_list = [identity for i in range(n_qubits)]

            # skip over identity terms
            identity_coeficient = 0
            if len(term) == 0:
                identity_coeficient = coef
                continue

            # contruct correct operator
            for qubit, pauli in term:

                if pauli == 'X':
                    full_op_list[qubit] = pauli_x

                if pauli == 'Y':
                    full_op_list[qubit] = pauli_y

                if pauli == 'Z':
                    full_op_list[qubit] = pauli_z


            # put into matrix form (tensor product)
            if len(full_op_list) == 1:
                full_op = full_op_list[0]
            else:
                full_op = np.kron(full_op_list[0], full_op_list[1])
                for i in range(2, len(full_op_list)):
                    full_op = np.kron(full_op, full_op_list[i])


            # caculates expectation for term
            op_on_state = full_op.dot(final_state)

            inner_product = final_state.conjugate().dot(op_on_state)

            expectation += inner_product * coef

        expectation += identity_coeficient
        return(expectation.real)

    elif method == 'sampling':
        pass

    else:
        raise ValueError('expectation is either wavefunction or sampling')
