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

"""Parameterized circuits and results."""

from cirq.study.flatten_expressions import (
    ExpressionMap,
    flatten,
    flatten_with_params,
    flatten_with_sweep,
)

from cirq.study.resolver import (
    ParamDictType,
    ParamMappingType,
    ParamResolver,
    ParamResolverOrSimilarType,
)

from cirq.study.sweepable import Sweepable, to_resolvers, to_sweep, to_sweeps

from cirq.study.sweeps import (
    Linspace,
    ListSweep,
    Points,
    Product,
    Sweep,
    UnitSweep,
    Zip,
    ZipLongest,
    dict_to_product_sweep,
    dict_to_zip_sweep,
)

from cirq.study.result import ResultDict, Result
