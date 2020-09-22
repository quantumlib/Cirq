# Copyright 2020 The Cirq Developers
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

from typing import Any

import cirq


def assert_consistent_resolve_parameters(val: Any):
    names = cirq.parameter_names(val)
    symbols = cirq.parameter_symbols(val)

    assert {symbol.name for symbol in symbols} == names

    if not cirq.is_parameterized(val):
        assert not names
        assert not symbols
    else:
        # Try to resolve all parameters with numbers. This may fail if some of
        # the parameters want different types. But if resolution succeeds, the
        # object should report that it has no more parameters to resolve.
        try:
            resolved = cirq.resolve_parameters(val, {name: 0 for name in names})
        except Exception:
            pass
        else:
            assert not cirq.parameter_names(resolved)
            assert not cirq.parameter_symbols(resolved)
