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

import cirq


def test_compute_displays_result_eq():
    eq = cirq.testing.EqualsTester()

    u = cirq.ComputeDisplaysResult(
        params=cirq.ParamResolver({'a': 2}),
        display_values={'k': 1.0})
    v = cirq.ComputeDisplaysResult(
        params=cirq.ParamResolver({'a': 2}),
        display_values={'k': 1.0})
    w = cirq.ComputeDisplaysResult(
        params=cirq.ParamResolver({'b': 2}),
        display_values={'k': 1.0})
    x = cirq.TrialResult.from_single_parameter_set(
        params=cirq.ParamResolver({'a': 2}),
        measurements={'k': np.array([[1]])})

    eq.add_equality_group(u, v)
    eq.add_equality_group(w)
    eq.add_equality_group(x)


def test_compute_displays_result_repr():
    v = cirq.ComputeDisplaysResult(
        params=cirq.ParamResolver({'a': 2}),
        display_values={'k': 1.0})

    cirq.testing.assert_equivalent_repr(v)
