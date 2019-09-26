# Copyright 2018 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import sympy

from cirq.contrib.quirk.cells.parse import parse_matrix, parse_formula


def test_parse_matrix():
    s = np.sqrt(0.5)
    np.testing.assert_allclose(parse_matrix('{{√½,√½},{-√½,√½}}'),
                               np.array([[s, s], [-s, s]]),
                               atol=1e-8)
    np.testing.assert_allclose(parse_matrix('{{√½,√½i},{√½i,√½}}'),
                               np.array([[s, s * 1j], [s * 1j, s]]),
                               atol=1e-8)
    np.testing.assert_allclose(parse_matrix('{{1,-i},{i,1+i}}'),
                               np.array([[1, -1j], [1j, 1 + 1j]]),
                               atol=1e-8)


def test_parse_formula():
    t = sympy.Symbol('t')
    assert parse_formula('t*t + ln(t)') == t * t + sympy.ln(t)
    assert parse_formula('cos(pi*t)') == sympy.cos(sympy.pi * t)
    np.testing.assert_allclose(parse_formula('cos(pi)'), -1, atol=1e-8)
    assert type(parse_formula('cos(pi)')) is float
