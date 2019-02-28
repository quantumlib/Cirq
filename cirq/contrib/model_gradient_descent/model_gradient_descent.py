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

from typing import Callable, List, Tuple

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def _get_least_squares_model_gradient(
        xs: List[np.ndarray],
        ys: List[np.ndarray],
        xopt: np.ndarray,
        linear_model: LinearRegression):
    model = Pipeline([('poly', PolynomialFeatures(degree=2)),
                      ('linear_model', linear_model),
                      ]
                     )
    shifted_xs = [x - xopt for x in xs]
    model = model.fit(shifted_xs, ys)
    fitted_coeffs = model.named_steps['linear_model'].coef_
    n = len(xs[0])
    linear_coeffs = np.array(fitted_coeffs[1:n+1])
    return linear_coeffs


def _random_point_in_ball(n: int, radius: float):
    point_on_sphere = np.random.randn(n)
    point_on_sphere /= np.linalg.norm(point_on_sphere)
    length = np.random.uniform()
    length = radius * length**(1/n)
    return length * point_on_sphere


def model_gradient_descent(
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        sample_radius: float=1e-1,
        n_sample_points: int=100,
        rate: float=1e-1,
        tol: float=1e-8,
        known_values: Tuple[List[np.ndarray], List[np.ndarray]]=None,
        max_evaluations: int=None,
        verbose: bool=False):
    """Model gradient descent algorithm for black-box optimization.

    The idea of this algorithm is to perform gradient descent, but estimate
    the gradient using a surrogate model instead of, say, by
    finite-differencing. The surrogate model is a least-squared quadratic
    fit to points sampled from the vicinity of the current iterate.
    This algorithm works well when you have an initial guess which is in the
    convex neighborhood of a local optimum and you want to converge to that
    local optimum. It's meant to be used when the function is stochastic.

    Args:
        f: The function to minimize.
        x0: An initial guess.
        sample_radius: The radius around the current iterate to sample
            points from to build the quadratic model.
        n_sample_points: The number of points to sample in each iteration.
        rate: The learning rate for the gradient descent.
        tol: The algorithm terminates when the difference between the current
            iterate and the next suggested iterate is smaller than this value.
        known_values: Any prior known values of the objective function.
            This is given as a tuple where the first element is a list
            of points and the second element is a list of the function values
            at those points.
        max_evaluations: The maximum number of function evaluations to allow
            before termination.
        verbose: Whether to print some information as the optimization proceeds.
    """

    if known_values is not None:
        known_xs, known_ys = known_values
    else:
        known_xs, known_ys = [], []

    current_x = x0
    total_evals = 0
    n = len(current_x)

    while max_evaluations is None or total_evals < max_evaluations:
        # Determine points to evaluate
        new_xs = [current_x + _random_point_in_ball(n, sample_radius)
                  for _ in range(n_sample_points)]
        if max_evaluations and total_evals + len(new_xs) > max_evaluations:
            break
        # Evaluate points
        new_ys = [f(x) for x in new_xs]
        total_evals += len(new_ys)
        known_xs.extend(new_xs)
        known_ys.extend(new_ys)
        # Determine points to use to build model
        model_xs = []
        model_ys = []
        for x, y in zip(known_xs, known_ys):
            if np.linalg.norm(x - current_x) < sample_radius:
                model_xs.append(x)
                model_ys.append(y)
        # Build and solve model
        linear_model = LinearRegression(fit_intercept=False)
        model_gradient = _get_least_squares_model_gradient(
            model_xs,
            model_ys,
            current_x,
            linear_model
        )
        gradient_norm = np.linalg.norm(model_gradient)
        # Print some info
        if verbose:
            print('Total evals: {}'.format(total_evals))
            print('# Points used for trust region model: {}'.format(
                len(model_xs)))
            print('2-norm of model gradient: {}'.format(gradient_norm))
            print()
        # Convergence criteria
        if rate*gradient_norm < tol:
            break
        # Update
        current_x -= rate*model_gradient

    return current_x, f(current_x), total_evals
