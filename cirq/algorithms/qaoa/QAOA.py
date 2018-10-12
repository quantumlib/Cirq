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

import numpy as np
import pyswarm

from scipy.optimize import minimize
from cirq.google import XmonSimulator, XmonMeasurementGate
from cirq.circuits import Circuit
from cirq.ops import RotXGate, RotYGate, MeasurementGate, H, Pauli, PauliString
from cirq.devices import GridQubit
from cirq.line import LineQubit
from exponentiate import exponentiate_qubit_operator
from expectation_value import expectation_value, expectation_from_sampling
from bayes_opt import BayesianOptimization


class QAOA:
    """
    A class for the quantum approximate optimization algorithm
        based on https://arxiv.org/pdf/1411.4028.pdf

    """

    def __init__(self, n_qubits, objective_function, mixing_operator='X'):
        """
        Initializes the QAOA object

        Args:
            n_qubits: number of qubits used in the circuit.
            objective_function: Dict[PauliString: Coefficient]
                the objective function QAOA wants to maximize.
            mixing_operator: Union[Dict[PauliString: Coefficient], str]
                This is the B operator in the original QAOA paper. It is used
                to compute the unitary U_B through exponentiating
                U_B = exp(i*\beta*B).
                Default set to 'X' which sets it to pauli X on every qubit.
        """

        self.objective_function = objective_function

        # setup cirq circuit and device
        self.circuit = Circuit()
        self.qubits = [LineQubit(i) for i in range(n_qubits)]

        # set mixing, or use standard if 'X'
        if mixing_operator == 'X':
            self.set_mixing_operator()
        else:
            self.mixing_operator = mixing_operator


    def set_objective_function(self, objective_function):
        """
        sets new objective_function

        Args:
            objective_function: the objective function QAOA wants to maximize.
        """
        self.objective_function = objective_function

    def set_mixing_operator(self, mixing_operator=None):
        """
        sets new mixing hermitian operator (B in QAOA paper) that will be
        used to construct the unitary U_B(\beta) = Exp(-i*beta*B)

        Args:
            mixing_operator: Dict[PauliString: Coefficient]
                operator B in QAOA paper, used to create unitary by
                exponentiation default set to 'X' which sets it to pauli X on
                every qubit.
        """

        # initialize operator dict
        self.mixing_operator = {}

        # sets it to X on every qubit
        if mixing_operator == None:
            # X on every qubit
            for q in self.qubits:
                self.mixing_operator[PauliString(
                    qubit_pauli_map={q: Pauli.X})] = 1

        # sets it to argument given in function
        else:
            self.mixing_operator = mixing_operator

    def regular_tree_max_cut_graph(self, n_connections):
        """
        sets up the cost function for a regular tree graph (no loops) where
        each vertex has n_connections edges connected to it.

        Args:
            n_connections: integer determining the number of connections
                for each vertex in the regular tree graph.
        """
        z = Pauli.Z

        # Ignores constant term since it just adds a phase
        self.objective_function = dict()

        # connect 0 to 1:
        self.objective_function[PauliString(qubit_pauli_map={self.qubits[0]: z,
                                            self.qubits[1]: z})] = -1/2

        # self.objective_function[()] = 1 / 2
        self.objective_function[()] = (len(self.qubits) - 1) / 2

        # connections
        # odd: (n_connections-1)*i + (4-n_connections + 2*j)
        # even: (n_connections-1)*i + 2 + 2*j
        for i in range(len(self.qubits)):
            for j in range(n_connections-1):

                if i % 2 == 1:
                    odd_connection = (n_connections-1)*i + (4 -
                                                            n_connections + 2*j)
                    if odd_connection < len(self.qubits):
                        self.objective_function[PauliString(
                            qubit_pauli_map={self.qubits[i]: z,
                                            self.qubits[odd_connection]:
                                                z})] = -1/2

                # even qubits
                elif i % 2 == 0:
                    even_connection = (n_connections-1)*i + 2 + 2*j
                    if even_connection < len(self.qubits):
                        self.objective_function[PauliString(
                            qubit_pauli_map={self.qubits[i]: z,
                                             self.qubits[even_connection]:
                                                 z})] = -1/2

    def setup_circuit(self, p, params):

        """
        Constructs circuit to generate state |beta,gamma> U_B U_C .... |s>

        Args:
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            params: list with values of gammas and betas, gammas
                are the entries params[0:p] and betas are params[p:2p].
        """

        # initial uniform superposition
        self.circuit.append(H.on_each(self.qubits))

        # loop over p iterations
        for p_val in range(p):
            gamma = params[p_val]
            beta = params[p+p_val]

            # calculates exponentiation of cost with appropriate gamma
            U_C = exponentiate_qubit_operator(time=gamma,
                                              operator=self.objective_function,
                                              trotter_steps=1)

            # appends to circuit
            self.circuit += U_C

            # calculates exponentiation of mixing operator B
            # with beta coefficient
            U_B = exponentiate_qubit_operator(time=beta,
                                              operator=self.mixing_operator,
                                              trotter_steps=1)

            # appends to circuit
            self.circuit += U_B
            # print(self.circuit)

    def run_circuit(self, params, p,
                    expectation_method='wavefunction',
                    min_flag=False):
        """
        Sets up circuit and returns expectation value

        Args:
            params: list with values of gammas and betas, gammas are the
                entries params[0:p] and betas are params[p:2p].
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            expectation_method: determines whether expectation value
                is obtained by an inner product with the final state or
                by sampling from measurements given repeated runs from the
                circuit if sampling is desired, enter int corresponding to the
                number of samples used to get expectation value.
            min_flag: bool that determines whether to return negative of cost.

        Returns:
            expectation value of self.objective_function in state prepared
                by circuit created using params.
        """

        # clears old circuit
        self.circuit = Circuit()

        # new circuit to prepare QAOA state with appropriate params
        self.setup_circuit(p, params)

        # calculates expectation value in state according to method
        if expectation_method == 'wavefunction':
            cost = expectation_value(circuit=self.circuit,
                                     operator=self.objective_function)

        elif isinstance(expectation_method, int):
            cost = expectation_from_sampling(circuit=self.circuit,
                                             operator=self.objective_function,
                                             n_samples=expectation_method)

        else:
            raise TypeError('method must be wavefunction or an '
                            'integer representing number of samples')

        # default is maximizing objectvie function, if flag is passed,
        # multiplies it by -1
        if min_flag:
            return -1*cost

        return cost

    def return_configuration_from_circuit(self, params, p,
                                          sampling=0):
        """
        Sets up circuit and returns expectation value

        Args:
            params: list with values of gammas and betas, gammas are the
                entries params[0:p] and betas are params[p:2p].
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            sampling: int determining whether we are sampling from the circuit
                or returning the result which has the largest probability
                amplitude

        Returns:
            dict with results of measurements or probability amplitudes
             of QAOA runs
        """
        measurement_dict = {}

        # clears old circuit
        self.circuit = Circuit()

        # new circuit to prepare QAOA state with appropriate params
        self.setup_circuit(p, params)

        # setup simulator
        sim = XmonSimulator()

        # calculates expectation value in state according to method
        if sampling == 0:

            # run simulator
            sim_result = sim.simulate(self.circuit)

            # get final state and norm
            final_state_prob = np.absolute(sim_result.final_state)**2
            max_prob_index = np.argmax(final_state_prob)

            # list with bits
            bits = [bin(i)[2:].zfill(n_qubits) for i in range(2 ** n_qubits)]

            # return probability of measurement and the bitstring that
            # corresponds to it
            measurement_dict['max_probability'] = final_state_prob[
                max_prob_index]

            measurement_dict['bits'] = bits[max_prob_index]

            return measurement_dict

        elif isinstance(sampling, int):

            # append measurements to circuit:
            meas = [XmonMeasurementGate(key=index).on(
                self.qubits[index]) for index in range(len(self.qubits))]

            self.circuit.append(meas)

            # run simulator
            sim_result = sim.run(circuit=self.circuit,
                                 repetitions=sampling)
            # measurement keys
            keys = [gate.key for _, _, gate in
                    circuit.findall_operations_with_gate_type(
                        XmonMeasurementGate)]

            # histogram of measurements
            counter = sim_result.multi_measurement_histogram(keys=keys)

            # most common measurement
            most_common = counter.most_common()[0]

            # most common measurement dict:
            measurement_dict['most_common_measurement'] = most_common[0]
            measurement_dict['most_common_measurement_number'] = most_common[1]
            measurement_dict['number_measurements'] = sampling

            return measurement_dict

    def optimization(self, p,
                     initial_params=None,
                     optimizer='swarm',
                     expectation_method='wavefunction'):
        """
        Main function that optimizes the parameters for QAOA

        Args:
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            initial_params: set initialization parameters if allowed by
                optimizer.
            optimizer: string that sets the optimization method used to
                obtain the parameters gamma and beta that maximize the objective
                function. Currently supports:
                    swarm - swarming algorithm
                    NM - Nelder-Mead
                    bayesian - Bayesian optimization
            expectation_method: determines whether expectation value
                is obtained by an inner product with the final state or
                by sampling from measurements given repeated runs from the
                circuit if sampling is desired, enter int corresponding to the
                number of samples used to get expectation value.

        Returns:
            returns a dictionary of the solution found using the given
                optimization method.
                keys:
                cost - final value of the objective function we are maximizing.
                gammas - list containing the p optimal gammas.
                betas - list containing the p  optimal betas.
        """

        if optimizer == 'swarm':

            return self.swarming_optimization(p, expectation_method)

        elif optimizer == 'NM':

            return self.nelder_mead_optimization(p, initial_params,
                                                 expectation_method)

        elif optimizer == 'bayesian':

            return self.bayesian_optimization(p, expectation_method)

        else:
            raise TypeError('Optimizer not supported')

    def swarming_optimization(self, p,
                              expectation_method='wavefunction'):
        """
        Main function that optimizes the parameters for QAOA

        Args:
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            expectation_method: determines whether expectation value
                is obtained by an inner product with the final state or
                by sampling from measurements given repeated runs from the
                circuit if sampling is desired, enter int corresponding to the
                number of samples used to get expectation value.

        Returns:
            returns a dictionary of the solution found using the given
                optimization method.
                keys:
                cost - final value of the objective function we are maximizing.
                gammas - list containing the p optimal gammas.
                betas - list containing the p  optimal betas.
        """

        # initialize solution dict
        solution = {}

        # lower and upper bounds for parameters in swarming optimization
        lower_bound = np.zeros(2 * p)
        upper_bound = np.concatenate((np.ones(p) * (2 * np.pi),
                                      np.ones(p) * np.pi))

        # flag needed since pyswarm finds min
        min_flag = True

        # calls swarming algorithm
        xopt, fopt = pyswarm.pso(self.run_circuit, lower_bound, upper_bound,
                                 f_ieqcons=None,
                                 args=(p, expectation_method, min_flag),
                                 swarmsize=20)

        # constructs solution dict
        solution['cost'] = -1*fopt
        solution['gammas'] = xopt[:p]
        solution['betas'] = xopt[p:]

        return solution

    def nelder_mead_optimization(self, p, initial_params,
                                 expectation_method='wavefunction'):
        """
        Main function that optimizes the parameters for QAOA

        Args:
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            initial_params: set initialization parameters if allowed by
                optimizer.
            expectation_method: determines whether expectation value
                is obtained by an inner product with the final state or
                by sampling from measurements given repeated runs from the
                circuit if sampling is desired, enter int corresponding to the
                number of samples used to get expectation value.

        Returns:
            returns a dictionary of the solution found using the given
                optimization method.
                keys:
                cost - final value of the objective function we are maximizing.
                gammas - list containing the p optimal gammas.
                betas - list containing the p  optimal betas.
        """

        # initialize solution dict
        solution = {}

        # initial parameters in optimization
        if initial_params is None:
            initial_gammas = np.random.random(p) * 2 * np.pi
            initial_betas = np.random.random(p) * np.pi
            initial_params = np.concatenate((initial_gammas, initial_betas))

        # flag needed since scipy.optimize.minimize finds min
        min_flag = True

        # call optimization routine
        result = minimize(fun=self.run_circuit,
                          x0=initial_params,
                          args=(p, expectation_method, min_flag),
                          method='Nelder-Mead')

        # constructs solution dict
        solution['cost'] = -1 * result.fun
        solution['gammas'] = result.x[:p]
        solution['betas'] = result.x[p:]

        return solution

    def bayesian_optimization(self, p,
                              expectation_method='wavefunction'):
        """
        Main function that optimizes the parameters for QAOA

        Args:
            p: number of repetitions of the unitaries U_C and U_B used in
                the state preparation in QAOA. Since each time a unitary is
                applied we introduce a new parameter, there are 2*p total
                parameters we want to optimize.
            expectation_method: determines whether expectation value
                is obtained by an inner product with the final state or
                by sampling from measurements given repeated runs from the
                circuit if sampling is desired, enter int corresponding to the
                number of samples used to get expectation value.

        Returns:
            returns a dictionary of the solution found using the given
                optimization method.
                keys:
                cost - final value of the objective function we are maximizing.
                gammas - list containing the p optimal gammas.
                betas - list containing the p  optimal betas.
        """

        # initialize solution dict
        solution = {}

        # helper function to unpack parameters
        def bayesian_run_custom(pval=p,
                                expectation=expectation_method,
                                **kwargs):

            p = pval
            params = [kwargs['gamma_{}'.format(m)] for m in range(p)]
            expectation_method = expectation
            # print(params)
            for i in range(p):
                params.append(kwargs['beta_{}'.format(i)])

            return self.run_circuit(params, p,
                                    expectation_method=expectation_method,
                                    min_flag=False)

        # parameter dict needed for bayesian optimization
        args_dict = dict()
        # args_dict['p'] = p

        # set up bounds for gammas and betas
        for i in range(p):
            args_dict['gamma_{}'.format(i)] = (0, 2 * np.pi)
            args_dict['beta_{}'.format(i)] = (0, np.pi)

        # initialize the optimizer
        boptimizer = BayesianOptimization(bayesian_run_custom, args_dict)

        # call maximization method
        boptimizer.maximize()

        # constructs solution dict
        solution['cost'] = boptimizer.Y.max()
        solution['gammas'] = boptimizer.X[:, :p]
        solution['betas'] = boptimizer.X[:, p:]

        return solution
