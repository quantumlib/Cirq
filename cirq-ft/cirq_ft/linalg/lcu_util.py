# Copyright 2023 The Cirq Developers
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

"""Utility methods for LCU circuits as implemented in https://github.com/quantumlib/OpenFermion"""

import math


def _partial_sums(vals):
    """Adds up the items in the input, yielding partial sums along the way."""
    total = 0
    for v in vals:
        yield total
        total += v
    yield total


def _differences(weights):
    """Iterates over the input yielding differences between adjacent items."""
    previous_weight = None
    have_previous_weight = False
    for w in weights:
        if have_previous_weight:
            yield w - previous_weight
        previous_weight = w
        have_previous_weight = True


def _discretize_probability_distribution(unnormalized_probabilities, epsilon):
    """Approximates probabilities with integers over a common denominator.

    Args:
        unnormalized_probabilities: A list of non-negative floats proportional
            to probabilities from a probability distribution. The numbers may
            not be normalized (they do not have to add up to 1).
        epsilon: The absolute error tolerance.

    Returns:
        numerators (list[int]): A list of numerators for each probability.
        denominator (int): The common denominator to divide numerators by to
            get probabilities.
        sub_bit_precision (int): The exponent mu such that
            denominator = n * 2**mu
            where n = len(unnormalized_probabilities).

        It is guaranteed that numerators[i] / denominator is within epsilon of
        the i'th input probability (after normalization).
    """
    n = len(unnormalized_probabilities)
    sub_bit_precision = max(0, int(math.ceil(-math.log(epsilon * n, 2))))
    bin_count = 2**sub_bit_precision * n

    cumulative = list(_partial_sums(unnormalized_probabilities))
    total = cumulative[-1]
    discretized_cumulative = [int(math.floor(c / total * bin_count + 0.5)) for c in cumulative]
    discretized = list(_differences(discretized_cumulative))
    return discretized, bin_count, sub_bit_precision


def _preprocess_for_efficient_roulette_selection(discretized_probabilities):
    """Prepares data for performing efficient roulette selection.

    The output is a tuple (alternates, keep_weights). The output is guaranteed
    to satisfy a sampling-equivalence property. Specifically, the following
    sampling process is guaranteed to be equivalent to simply picking index i
    with probability weights[i] / sum(weights):

        1. Pick a number i in [0, len(weights) - 1] uniformly at random.
        2. Return i With probability keep_weights[i]*len(weights)/sum(weights).
        3. Otherwise return alternates[i].

    In other words, the output makes it possible to perform roulette selection
    while generating only two random numbers, doing a single lookup of the
    relevant (keep_chance, alternate) pair, and doing one comparison. This is
    not so useful classically, but in the context of a quantum computation
    where all those things are expensive the second sampling process is far
    superior.

    Args:
        discretized_probabilities: A list of probabilities approximated by
            integer numerators (with an implied common denominator). In order
            to operate without floating point error, it is required that the
            sum of this list is a multiple of the number of items in the list.

    Returns:
        alternates (list[int]): An alternate index for each index from 0 to
            len(weights) - 1
        keep_weight (list[int]): Indicates how often one should stay at index i
            instead of switching to alternates[i]. To get the actual keep
            probability of the i'th element, multiply keep_weight[i] by
            len(discretized_probabilities) then divide by
            sum(discretized_probabilities).
    Raises:
        ValueError: if `discretized_probabilities` input is empty or if the sum of elements
            in the list is not a multiple of the number of items in the list.
    """
    weights = list(discretized_probabilities)  # Need a copy we can mutate.
    if not weights:
        raise ValueError('Empty input.')

    n = len(weights)
    target_weight = sum(weights) // n
    if sum(weights) != n * target_weight:
        raise ValueError('sum(weights) must be a multiple of len(weights).')

    # Initially, every item's alternative is itself.
    alternates = list(range(n))
    keep_weights = [0] * n

    # Scan for needy items and donors. First pass will handle all
    # initially-needy items. Second pass will handle any remaining items that
    # started as donors but become needy due to over-donation (though some may
    # also be handled during the first pass).
    donor_position = 0
    for _ in range(2):
        for i in range(n):
            # Is this a needy item?
            if weights[i] >= target_weight:
                continue  # Nope.

            # Find a donor.
            while weights[donor_position] <= target_weight:
                donor_position += 1

            # Donate.
            donated = target_weight - weights[i]
            weights[donor_position] -= donated
            alternates[i] = donor_position
            keep_weights[i] = weights[i]

            # Needy item has been paired. Remove it from consideration.
            weights[i] = target_weight

    return alternates, keep_weights


def preprocess_lcu_coefficients_for_reversible_sampling(lcu_coefficients, epsilon):
    """Prepares data used to perform efficient reversible roulette selection.

    Treats the coefficients of unitaries in the linear combination of
    unitaries decomposition of the Hamiltonian as probabilities in order to
    decompose them into a list of alternate and keep numerators allowing for
    an efficient preparation method of a state where the computational basis
    state :math. `|k>` has an amplitude proportional to the coefficient.

    It is guaranteed that following the following sampling process will
    sample each index k with a probability within epsilon of
    lcu_coefficients[k] / sum(lcu_coefficients) and also,
    1. Uniformly sample an index i from [0, len(lcu_coefficients) - 1].
    2. With probability keep_numers[i] / by keep_denom, return i.
    3. Otherwise return alternates[i].

    Args:
        lcu_coefficients: A list of non-negative floats, with the i'th float
            corresponding to the i'th coefficient of an LCU decomposition
            of the Hamiltonian (in an ordering determined by the caller).
        epsilon: Absolute error tolerance.

    Returns:
        alternates (list[int]): A python list of ints indicating alternative
            indices that may be switched to after generating a uniform index.
            The int at offset k is the alternate to use when the initial index
            is k.
        keep_numers (list[int]): A python list of ints indicating the
            numerators of the probability that the alternative index should be
            used instead of the initial index.
        sub_bit_precision (int): A python int indicating the exponent of the
            denominator to divide the items in keep_numers by in order to get
            a probability. The actual denominator is 2**sub_bit_precision.
    """
    numers, denom, sub_bit_precision = _discretize_probability_distribution(
        lcu_coefficients, epsilon
    )
    assert denom == 2**sub_bit_precision * len(numers)
    alternates, keep_numers = _preprocess_for_efficient_roulette_selection(numers)
    return alternates, keep_numers, sub_bit_precision
