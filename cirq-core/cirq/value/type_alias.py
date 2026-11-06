# Copyright 2019 The Cirq Developers
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

from __future__ import annotations

import numpy as np
import sympy

from cirq._doc import document

"""Supply aliases for commonly used types.
"""

TParamKey = str | sympy.Expr
document(TParamKey, """A parameter that a parameter resolver may map to a value.""")

TParamVal = float | np.integer | np.floating | sympy.Expr
document(
    TParamVal,
    """A value that a parameter resolver may return for a parameter.

    This is a real float, a NumPy integer or floating-point number, or a sympy expression.

    NumPy fixed-width scalars are stored as given, so they keep NumPy's overflow and wraparound
    behavior in gate arithmetic. For example, an unsigned integer exponent can overflow in
    `cirq.inverse`.
    """,
)

TParamValComplex = complex | np.number | sympy.Expr
document(TParamValComplex, """A complex value that parameter resolvers may use for parameters.""")
