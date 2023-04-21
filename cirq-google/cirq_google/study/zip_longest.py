# Copyright 2023 The Cirq Developers
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
from typing import Any, Dict, Iterator, List

import itertools
import cirq


class ZipLongest(cirq.Zip):
    """Iterate over constituent sweeps in parallel

    Analogous to itertools.zip_longest.
    Note that we iterate until all sweeps terminate,
    so if the sweeps are different lengths, the
    shorter sweeps will be filled by repeating their last value
    until all sweeps have equal length.
    This is different from itertools.zip_longest, which uses a fixed fill value.
    """

    def __init__(self, *sweeps: cirq.Sweep) -> None:
        self.sweeps = sweeps

    def __eq__(self, other):
        if not isinstance(other, ZipLongest):
            return NotImplemented
        return self.sweeps == other.sweeps

    def __hash__(self) -> int:
        return hash(tuple(self.sweeps))

    @property
    def keys(self) -> List['cirq.TParamKey']:
        return sum((sweep.keys for sweep in self.sweeps), [])

    def __len__(self) -> int:
        if not self.sweeps:
            return 0
        return max(len(sweep) for sweep in self.sweeps)

    def __repr__(self) -> str:
        sweeps_repr = ', '.join(repr(s) for s in self.sweeps)
        return f'cirq_google.ZipLongest({sweeps_repr})'

    def __str__(self) -> str:
        sweeps_repr = ', '.join(repr(s) for s in self.sweeps)
        return f'ZipLongest({sweeps_repr})'

    def param_tuples(self) -> Iterator[cirq.study.sweeps.Params]:
        iters = [
            itertools.chain(sweep.param_tuples(), itertools.repeat(list(sweep.param_tuples())[-1]))
            for sweep in self.sweeps
        ]
        for vals in itertools.islice(zip(*iters), len(self)):
            yield sum(vals, ())

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.obj_to_dict_helper(self, ['sweeps'])

    @classmethod
    def _from_json_dict_(cls, sweeps, **kwargs):
        return ZipLongest(*sweeps)
