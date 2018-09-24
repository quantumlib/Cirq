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

"""
# QAOA - Quantum Approximate Optimization Algorithm

This is an implementation of the QAOA algorithm by
Farhi, Goldstone, and Gutmann.

Reference:

https://arxiv.org/abs/1411.4028

QAOA is an algorithm whose goal is to optimize an
objective function by using a quantum processor to
implement parametrized unitaries. The state that is
prepared by the application of these unitaries is used
to estimate the expectation value of the objective function.
The parameters and expectation value are then used as inputs
in a classical optimizer that outputs, according to
update rules set by the choice of optimization routine, a new
set of parameters that are fed back into the quantum processor
and restarts the loop. This continues until some stopping
criterion is met.


## Running example code

To run the code, modify the file qaoa_example.py and run the
script:
```buildoutcfg
python qaoa_example.py
```
Alternatively, to run a regular tree graph with number of
connections set by n_connections parameter, run
qaoa_function.py


## Code structure

The main code is in QAOA.py, which contains a QAOA
class and methods that are used to setup the algorithm.

Currently the supported optimizers are:

- Particle Swarm Optimization: 'swarm'
- Nelder-Mead: 'nelder-mead'
- Bayesian Optimization: 'bayesian'

Expectation value is estimated by default by accessing
the wavefunction and performing an inner-product. This
corresponds to the 'wavefunction' key to the
expectation_method argument. If sampling is desired
simply enter an integer corresponding to the number
of runs used to estimate the expectation value of the
objective function.
"""

from cirq.algorithms.qaoa.QAOA import (
    QAOA,
)

from cirq.algorithms.qaoa.qaoa_function.py import (
    qaoa_solver,
    qaoa_regular_graph,
)