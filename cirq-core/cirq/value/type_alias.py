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

from typing import Union

import sympy

from cirq._doc import document
from cirq.value import linear_dict

"""Supply aliases for commonly used types.
"""

TParamKey = Union[str, sympy.Expr]
document(TParamKey, """A parameter that a parameter resolver may map to a value.""")

TParamVal = Union[float, sympy.Expr]
document(TParamVal, """A value that a parameter resolver may return for a parameter.""")

TParamValComplex = Union[linear_dict.Scalar, sympy.Expr]
document(TParamValComplex, """A complex value that parameter resolvers may use for parameters.""")
