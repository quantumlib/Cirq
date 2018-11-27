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

import typing
import numpy as np
from cirq.algorithms import qaoa

from typing import Union

def qaoa_solver(n_qubits,
                objective_function,
                optimization_method,
                p,
                mixing = 'X', expectation_method = 'wavefunction'):
    """
    QAOA solver helper function

    Args:
    n_qubits: number of qubits
    objective_function: objective function to be optimized
        type should be Dict[PauliString: float]
    optimization_method: type of optimizer used, currently available are
        swarm, nelder-mead, bayesian
    p: number of gamma and beta parameters/number of repeated
        applications of unitaries from QAOA paper
    mixing mixing operator: 'B' in QAOA paper. If standard 'X' on every
        qubit, set to 'X', otherwise specify with type Dict[PauliString: float]

    Returns:
        Dict with solution and gamma and beta values for optimal solution
    """

    algo = qaoa.QAOA(n_qubits=n_qubits,
                     objective_function=objective_function,
                     mixing_operator=mixing)

    optimzation_dict = {'swarm': 'swarm',
                        'nelder-mead': 'nelder-mead',
                        'bayesian': 'bayesian'}

    return algo.optimization(p=p, initial_params=None,
                             optimizer=optimzation_dict[optimization_method],
                             expectation_method=expectation_method)


def qaoa_regular_graph(n_qubits: int,
                       n_connections: int,
                       p: int, optimization_method: str,
                       expectation_method: Union[str, int] = 'wavefunction'):
    """
    Runs QAOA on a regular n_connections graph

    Args:
    n_qubits: number of qubits
    n_connections: number of connections for each vertex on regular graph
    p: number of gamma and beta parameters/number of repeated
        applications of unitaries from QAOA paper
    optimization_method: type of optimizer used, currently available are
        swarm, nelder-mead, bayesian
    expectation_method: whether objective function is calculated using
        the final state on the simulator or sampling by measurement
        set to 'wavefunction' for final state
        set to an integer for sampling, integer corresponds to number of samples

    Returns:
        dict with solution and gamma and beta values for optimal solution
    """

    algo = qaoa.qaoa_regular_tree_max_cut(n_vertices=n_qubits,
                                     degree=n_connections)

    optimzation_dict = {'swarm':'swarm',
                        'nelder-mead':'nelder-mead',
                        'bayesian':'bayesian'}

    return algo.optimization(p=p,
                             initial_params=None,
                             optimizer=optimzation_dict[optimization_method],
                             expectation_method=expectation_method)

