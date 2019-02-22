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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline


def get_least_squares_model_gradient(xs, ys, xopt, linear_model):
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


def model_gradient_descent(
        f,
        x0,
        sample_radius,
        rate,
        distance_function,
        sampling_function,
        tol=1e-7,
        adapt_sampling_radius=False,
        known_values=None,
        max_evaluations=None,
        true_f=None,
        verbose=False):
    if known_values is not None:
        known_xs, known_ys = known_values
    else:
        known_xs, known_ys = [], []
    current_x = x0
    total_evals = 0

    sample_radius_coefficient = None

    while max_evaluations is None or total_evals < max_evaluations:
        # Determine points to evaluate
        new_xs = sampling_function(current_x, sample_radius, known_xs, known_ys)
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
            if distance_function(x, current_x) < sample_radius:
                model_xs.append(x)
                model_ys.append(y)
        # Build and solve model
        linear_model = LinearRegression(fit_intercept=False)
        model_gradient = get_least_squares_model_gradient(
            model_xs,
            model_ys,
            current_x,
            linear_model
        )
        gradient_norm = np.linalg.norm(model_gradient)
        if sample_radius_coefficient is None:
            sample_radius_coefficient = sample_radius / gradient_norm

        # Print some info
        if verbose:
            print('Total evals: {}'.format(total_evals))
            if true_f is not None:
                print('True objective value: {}'.format(true_f(current_x)))
            print('# Points used for trust region model: {}'.format(len(model_xs)))
            print('2-norm of model gradient: {}'.format(gradient_norm))
            print()

        # Convergence criteria
        if rate*np.linalg.norm(model_gradient) < tol:
            break
        current_x -= rate*model_gradient

        if adapt_sampling_radius:
            sample_radius = sample_radius_coefficient * gradient_norm

    return current_x, f(current_x), total_evals
