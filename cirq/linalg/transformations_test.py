# Copyright 2018 Google LLC
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

from cirq.linalg.transformations import reflection_matrix_pow


def test_reflection_matrix_pow_consistent_results():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = reflection_matrix_pow(x, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_x, sqrt_x), x, atol=1e-10)

    ix = x * np.sqrt(1j)
    sqrt_ix = reflection_matrix_pow(ix, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_ix, sqrt_ix), ix, atol=1e-10)

    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5)
    cube_root_h = reflection_matrix_pow(h, 1/3)
    np.testing.assert_allclose(
        np.dot(np.dot(cube_root_h, cube_root_h), cube_root_h),
        h,
        atol=1e-8)

    y = np.array([[0, -1j], [1j, 0]])
    h = np.array([[1, 1], [1, -1]]) * np.sqrt(0.5j)
    yh = np.kron(y, h)
    sqrt_yh = reflection_matrix_pow(yh, 0.5)
    np.testing.assert_allclose(np.dot(sqrt_yh, sqrt_yh), yh, atol=1e-10)


def test_reflection_matrix_sign_preference_under_perturbation():
    x = np.array([[0, 1], [1, 0]])
    sqrt_x = np.array([[1, -1j], [-1j, 1]]) * (1 + 1j) / 2
    np.testing.assert_allclose(reflection_matrix_pow(x, 0.5),
                               sqrt_x,
                               atol=1e-8)

    # Sqrt should behave well when phased by less than 90 degrees.
    # (When rotating by more it's ambiguous. For example, 181 = 91*2 = -89*2.)
    for perturbation in [0, 0.1, -0.1, 0.3, -0.3, 0.49, -0.49]:
        px = x * complex(-1)**perturbation
        expected_sqrt_px = sqrt_x * complex(-1)**(perturbation / 2)
        sqrt_px = reflection_matrix_pow(px, 0.5)
        np.testing.assert_allclose(np.dot(sqrt_px, sqrt_px), px, atol=1e-10)
        np.testing.assert_allclose(sqrt_px, expected_sqrt_px, atol=1e-10)
