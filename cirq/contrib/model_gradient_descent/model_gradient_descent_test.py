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

from cirq.contrib.model_gradient_descent import model_gradient_descent


def sum_of_squares(x):
    return np.sum(x**2)


def test_model_gradient_descent():
    x0 = np.random.randn(10)
    sample_radius = 1e-1
    rate = 1e-1
    result = model_gradient_descent(
        sum_of_squares,
        x0,
        sample_radius=sample_radius,
        n_sample_points=10,
        rate=rate,
        tol=1e-8,
        known_values=None,
        max_evaluations=None,
        verbose=True)

    assert np.allclose(result.x, np.zeros(len(result.x)), atol=1e-7)
    assert np.allclose(result.fun, 0)
    assert isinstance(result.nfev, int)


def test_model_gradient_descent_with_known_values():
    x0 = np.random.randn(10)
    sample_radius = 1e-1
    rate = 1e-1
    known_xs = [np.ones(10)]
    known_ys = [10.0]
    result = model_gradient_descent(
        sum_of_squares,
        x0,
        sample_radius=sample_radius,
        n_sample_points=10,
        rate=rate,
        tol=1e-8,
        known_values=(known_xs, known_ys),
        max_evaluations=None,
        verbose=True)

    assert len(known_xs) == 1
    assert len(known_ys) == 1


def test_model_gradient_descent_limited_evaluations():
    x0 = np.random.randn(10)
    sample_radius = 1e-1
    rate = 1e-1
    result = model_gradient_descent(
        sum_of_squares,
        x0,
        sample_radius=sample_radius,
        n_sample_points=10,
        rate=rate,
        tol=1e-8,
        known_values=None,
        max_evaluations=20,
        verbose=True)

    assert isinstance(result.x, np.ndarray)
    assert isinstance(result.fun, float)
    assert result.nfev == 20
