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
from typing import Optional, Mapping, TYPE_CHECKING

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
            job_finished_time: A timestamp for when the job finished.
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
