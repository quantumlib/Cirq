# Copyright 2020 The Cirq Developers
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

from typing import Sequence

import numpy as np


def purity_from_probabilities(
    hilbert_space_dimension: int, probabilities: Sequence[float]
) -> float:
    """Purity estimator from speckle purity benchmarking.

    Estimates purity from empirical probabilities of observed bitstrings.

    This estimator assumes that the circuit used in experiment is sufficiently
    scrambling that its output probabilities follow the Porter-Thomas
    distribution. This assumption holds for typical instances of random quantum
    circuits of sufficient depth.

    The state resulting from the experimental implementation of the circuit
    is modeled as

        ρ = p |𝜓⟩⟨𝜓|  + (1 - p) I / D

    where |𝜓⟩ is a pure state, I / D is the maximally mixed state, and p is
    between 0 and 1. The purity of this state is given by p**2. If p = 1, then
    the bitstring probabilities are modeled as being drawn from the
    Porter-Thomas distribution, with probability density function given by

        f(x) = (D - 1) (1 - x)**(D - 2).

    The mean of this distribution is 1 / D and its variance is
    (D - 1) / [D**2 (D + 1)]. In general, the variance of the distribution
    is multipled by p**2. Therefore, the purity can be computed by dividing
    the variance of the empirical probabilities by the Porter-Thomas
    variance (D - 1) / [D**2 (D + 1)].

    Args:
        hilbert_space_dimension: Dimension of the Hilbert space on which the
            quantum circuits acts.
        probabilities: Empirical probabilities of bitstrings observed in
            experiment.
    Returns:
        Estimate of the purity of the state resulting from the experimental
        implementation of a quantum circuit.
    """
    D = hilbert_space_dimension
    porter_thomas_variance = (D - 1) / (D + 1) / D**2
    return np.var(probabilities) / porter_thomas_variance
