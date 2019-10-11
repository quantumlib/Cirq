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
"""A helper for jobs that have been created on the Quantum Engine."""

import time

from typing import Dict, List, Optional, TYPE_CHECKING

from cirq import study
from cirq.google.engine import calibration

if TYPE_CHECKING:
    import cirq.google.engine.engine as engine

TERMINAL_STATES = ['SUCCESS', 'FAILURE', 'CANCELLED']


class EngineJob:
    """A job created via the Quantum Engine API.

    This job may be in a variety of states. It may be scheduling, it may be
    executing on a machine, or it may have entered a terminal state
    (either succeeding or failing).

    Attributes:
      job_config: The JobConfig used to create the job.
      job_resource_name: The full resource name of the engine job.
    """

    def __init__(self, job_config: 'engine.JobConfig', job: Dict,
                 engine: 'engine.Engine') -> None:
        """A job submitted to the engine.

        Args:
            job_config: The JobConfig used to create the job.
            job: A full Job Dict.
            engine: An Engine instance associated with the same project as the
                job.
        """
        self.job_config = job_config
        self._job = job
        self._engine = engine
        self.job_resource_name = job['name']
        self.program_id = self.job_resource_name.split('/jobs')[0]
        self._results: Optional[List[study.TrialResult]] = None

    def _refresh_job(self) -> Dict:
        if self._job['executionStatus']['state'] not in TERMINAL_STATES:
            self._job = self._engine.get_job(self.job_resource_name)
        return self._job

    def status(self) -> str:
        """Return the execution status of the job."""
        return self._refresh_job()['executionStatus']['state']

    def get_calibration(self) -> Optional[calibration.Calibration]:
        """Returns the recorded calibration at the time when the job was run, if
        one was captured, else None."""
        status = self._job['executionStatus']
        if (not 'calibrationName' in status): return None
        return self._engine.get_calibration(status['calibrationName'])

    def cancel(self) -> None:
        """Cancel the job."""
        self._engine.cancel_job(self.job_resource_name)

    def results(self) -> List[study.TrialResult]:
        """Returns the job results, blocking until the job is complete.
        """
        if not self._results:
            job = self._refresh_job()
            for _ in range(1000):
                if job['executionStatus']['state'] in TERMINAL_STATES:
                    break
                time.sleep(0.5)
                job = self._refresh_job()
            self._raise_on_failure(job)
            self._results = self._engine.get_job_results(self.job_resource_name)
        return self._results

    def _raise_on_failure(self, job: Dict) -> None:
        execution_status = job['executionStatus']
        state = execution_status['state']
        name = job['name']
        if state != 'SUCCESS':
            if state == 'FAILURE':
                processor = (execution_status['processorName'] if
                             'processorName' in execution_status else 'UNKNOWN')
                error_code = execution_status['failure']['errorCode']
                error_message = execution_status['failure']['errorMessage']
                raise RuntimeError(
                    "Job {} on processor {} failed. {}: {}".format(
                        name, processor, error_code, error_message))
            elif state in TERMINAL_STATES:
                raise RuntimeError('Job {} failed in state {}.'.format(
                    name, state))
            else:
                raise RuntimeError(
                    'Timed out waiting for results. Job {} is in state {}'.
                    format(name, state))

    def __iter__(self):
        return iter(self.results())

    def __str__(self):
        return str('EngineJob({})'.format(self.job_resource_name))
