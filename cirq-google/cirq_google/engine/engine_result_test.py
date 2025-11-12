# Copyright 2022 The Cirq Developers
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

import datetime
from collections.abc import Mapping
from typing import TYPE_CHECKING

import numpy as np

import cirq
import cirq_google as cg

if TYPE_CHECKING:
    import pandas as pd

_DT = datetime.datetime(2022, 4, 1, 1, 23, 45, tzinfo=datetime.timezone.utc)


def test_engine_result():
    res = cg.EngineResult(
        job_id='my_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )

    assert res.job_id == 'my_job_id'
    assert res.measurements['a'].shape == (2, 2)

    cirq.testing.assert_equivalent_repr(res, global_vals={'cirq_google': cg})


def test_engine_result_from_result_dict():
    res = cg.EngineResult(
        job_id='my_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )

    res2 = cirq.ResultDict(
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )
    assert res2 != res
    assert res != res2
    assert res == cg.EngineResult.from_result(res2, job_id='my_job_id')


def test_engine_result_eq():
    res1 = cg.EngineResult(
        job_id='my_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )
    res2 = cg.EngineResult(
        job_id='my_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )
    assert res1 == res2

    res3 = cg.EngineResult(
        job_id='my_other_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )
    assert res1 != res3


class MyResult(cirq.Result):
    @property
    def params(self) -> cirq.ParamResolver:
        return cirq.ParamResolver()

    @property
    def measurements(self) -> Mapping[str, np.ndarray]:
        return {'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])}

    @property
    def records(self) -> Mapping[str, np.ndarray]:
        return {k: v[:, np.newaxis, :] for k, v in self.measurements.items()}

    @property
    def data(self) -> pd.DataFrame:  # pragma: no cover
        return cirq.Result.dataframe_from_measurements(self.measurements)


def test_engine_result_from_result():
    res = cg.EngineResult(
        job_id='my_job_id',
        params=None,
        measurements={'a': np.array([[0, 0], [1, 1]]), 'b': np.array([[0, 0, 0], [1, 1, 1]])},
    )

    res2 = MyResult()
    assert res == cg.EngineResult.from_result(res2, job_id='my_job_id')
