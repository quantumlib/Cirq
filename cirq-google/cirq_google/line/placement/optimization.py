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

import math

from typing import Callable, Tuple, TypeVar

T = TypeVar('T')


def _accept(random_sample: float, cost_diff: float, temp: float) -> Tuple[bool, float]:
    """Calculates probability and draws if solution should be accepted.

    Based on exp(-Delta*E/T) formula.

    Args:
        random_sample: Uniformly distributed random number in the range [0, 1).
        cost_diff: Cost difference between new and previous solutions.
        temp: Current temperature.

    Returns:
        Tuple of boolean and float, with boolean equal to True if solution is
        accepted, and False otherwise. The float value is acceptance
        probability.
    """
    exponent = -cost_diff / temp
    if exponent >= 0.0:
        return True, 1.0

    probability = math.exp(exponent)
    return probability > random_sample, probability


def anneal_minimize(
    initial: T,
    cost_func: Callable[[T], float],
    move_func: Callable[[T], T],
    random_sample: Callable[[], float],
    temp_initial: float = 1.0e-2,
    temp_final: float = 1e-6,
    cooling_factor: float = 0.99,
    repeat: int = 100,
    trace_func: Callable[[T, float, float, float, bool], None] = None,
) -> T:
    """Minimize solution using Simulated Annealing meta-heuristic.

    Args:
        initial: Initial solution of type T to the problem.
        cost_func: Callable which takes current solution of type T, evaluates it
            and returns float with the cost estimate. The better solution is,
            the lower resulting value should be; negative values are allowed.
        move_func: Callable which takes current solution of type T and returns a
            new solution candidate of type T which is random iteration over
            input solution. The input solution, which is argument to this
            callback should not be mutated.
        random_sample: Callable which gives uniformly sampled random value from
            the [0, 1) interval on each call.
        temp_initial: Optional initial temperature for simulated annealing
            optimization. Scale of this value is cost_func-dependent.
        temp_final: Optional final temperature for simulated annealing
            optimization, where search should be stopped. Scale of this value is
            cost_func-dependent.
        cooling_factor: Optional factor to be applied to the current temperature
            and give the new temperature, this must be strictly greater than 0
            and strictly lower than 1.
        repeat: Optional number of iterations to perform at each given
            temperature.
        trace_func: Optional callback for tracing simulated annealing progress.
            This is going to be called at each algorithm step for the arguments:
            solution candidate (T), current temperature (float), candidate cost
            (float), probability of accepting candidate (float), and acceptance
            decision (boolean).

    Returns:
        The best solution found.

    Raises:
        ValueError: When supplied arguments are invalid.
    """

    if not 0.0 < cooling_factor < 1.0:
        raise ValueError("Cooling factor must be within (0, 1) range")

    temp = temp_initial
    sol = initial
    sol_cost = cost_func(initial)
    best = sol
    best_cost = sol_cost

    if trace_func:
        trace_func(sol, temp, sol_cost, 1.0, True)

    while temp > temp_final:
        for _ in range(0, repeat):
            # Find a new solution candidate and evaluate its cost.
            cand = move_func(sol)
            cand_cost = cost_func(cand)

            # Store the best solution, regardless if it is accepted or not.
            if best_cost > cand_cost:
                best = cand
                best_cost = cand_cost

            accepted, probability = _accept(random_sample(), cand_cost - sol_cost, temp)
            if accepted:
                sol = cand
                sol_cost = cand_cost

            if trace_func:
                trace_func(cand, temp, cand_cost, probability, accepted)

        temp *= cooling_factor

    return best
