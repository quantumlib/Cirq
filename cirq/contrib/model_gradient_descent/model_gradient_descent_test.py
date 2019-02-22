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


def f(x):
    return np.sum(x**2)


def random_point_in_ball(n, radius):
    point_on_sphere = np.random.randn(n)
    point_on_sphere /= np.linalg.norm(point_on_sphere)
    length = np.random.uniform()
    length = radius * length**(1/n)
    return length * point_on_sphere


def sampling_function(current_x, sampling_radius, known_xs, known_ys):
    n_points = 300
    n = len(current_x)
    return [current_x + random_point_in_ball(n, sampling_radius) for _ in range(n_points)]


def distance_function(a, b):
    return np.linalg.norm(a - b)


x0 = np.random.randn(10)
sample_radius = 1e-1
rate = 1e-2


x, y, n_evals = model_gradient_descent(
    f,
    x0,
    sample_radius,
    rate=rate,
    distance_function=distance_function,
    sampling_function=sampling_function,
    tol=1e-10,
    adapt_sampling_radius=False,
    known_values=None,
    max_evaluations=None,
    true_f=None,
    verbose=True)
