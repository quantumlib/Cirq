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
from typing import Any

import pandas as pd

from cirq import protocols
from cirq._compat import proper_repr


class SampleResult:
    """The result of a `sampler.sample(...)` call."""

    def __init__(self, *, data: pd.DataFrame) -> None:
        """
        Args:
            data: A pandas `pd.DataFrame` where each row is a sample and there
                is a column for each parameter value and measurement result.
        """
        self.data = data

    def __eq__(self, other):
        if not isinstance(other, type(self)):
            return NotImplemented
        return self.data.equals(other.data)

    def __ne__(self, other):
        return not self == other

    def __str__(self):
        return f'SampleResult with data:\n{self.data}'

    def __repr__(self):
        return f'cirq.SampleResult(data={proper_repr(self.data)})'

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Text output in Jupyter."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('SampleResult(...)')
        else:
            p.text(str(self))

    def _json_dict_(self):
        return protocols.obj_to_dict_helper(self, ['data'])
