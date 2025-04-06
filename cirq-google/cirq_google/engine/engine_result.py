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
import datetime
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

import numpy as np

from cirq import study

if TYPE_CHECKING:
    import cirq


class EngineResult(study.ResultDict):
    """A ResultDict with additional job metadata.

    Please see the documentation for `cirq.ResultDict` for more information.

    Additional Attributes:
        job_id: A string job identifier.
        job_finished_time: A timestamp for when the job finished.
    """

    def __init__(
        self,
        *,  # Forces keyword args.
        job_id: str,
        job_finished_time: datetime.datetime,
        params: Optional[study.ParamResolver] = None,
        measurements: Optional[Mapping[str, np.ndarray]] = None,
        records: Optional[Mapping[str, np.ndarray]] = None,
    ):
        """Initialize the result.

        Args:
            job_id: A string job identifier.
            job_finished_time: A timestamp for when the job finished; will be converted to UTC.
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
            records: A dictionary from measurement gate key to measurement
                results. See `cirq.ResultDict`.
        """
        super().__init__(params=params, measurements=measurements, records=records)
        self.job_id = job_id
        self.job_finished_time = job_finished_time

    @classmethod
    def from_result(
        cls, result: 'cirq.Result', *, job_id: str, job_finished_time: datetime.datetime
    ):
        if isinstance(result, study.ResultDict):
            # optimize by using private methods
            return cls(
                params=result._params,
                measurements=result._measurements,
                records=result._records,
                job_id=job_id,
                job_finished_time=job_finished_time,
            )
        else:
            return cls(
                params=result.params,
                measurements=result.measurements,
                records=result.records,
                job_id=job_id,
                job_finished_time=job_finished_time,
            )

    def __eq__(self, other):
        if not isinstance(other, EngineResult):
            return False

        return (
            super().__eq__(other)
            and self.job_id == other.job_id
            and self.job_finished_time == other.job_finished_time
        )

    def __repr__(self) -> str:
        return (
            f'cirq_google.EngineResult(params={self.params!r}, '
            f'records={self._record_dict_repr()}, '
            f'job_id={self.job_id!r}, '
            f'job_finished_time={self.job_finished_time!r})'
        )

    @classmethod
    def _json_namespace_(cls) -> str:
        return 'cirq.google'

    def _json_dict_(self) -> Dict[str, Any]:
        d = super()._json_dict_()
        d['job_id'] = self.job_id
        d['job_finished_time'] = self.job_finished_time
        return d

    @classmethod
    def _from_json_dict_(cls, params, records, job_id, job_finished_time, **kwargs):
        return cls._from_packed_records(
            params=params, records=records, job_id=job_id, job_finished_time=job_finished_time
        )
