from typing import Union, List, TYPE_CHECKING

from cirq import work
from cirq.google.engine import engine

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


class QuantumEngineSampler(work.Sampler):
    """A sampler that samples from real hardware on the quantum engine."""

    def __init__(self,
                 *,
                 google_cloud_project_id: str,
                 processor_id: Union[str, List[str]],
                 gate_set: 'cirq.google.SerializableGateSet'):
        """
        Args:
            google_cloud_project_id: The identifier of the google cloud project
                that quantum engine is being used in.
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
        self.google_cloud_project_id = google_cloud_project_id
        self.processor = (
            [processor_id] if isinstance(processor_id, str) else processor_id
        )
        self.gate_set = gate_set

    def _request_builder(self, *args, **kwargs):
        from apiclient import http as apiclient_http
        request = apiclient_http.HttpRequest(*args, **kwargs)
        request.headers['X-Goog-User-Project'] = self.google_cloud_project_id
        return request

    def run_sweep(
            self,
            program: Union['cirq.Circuit', 'cirq.Schedule'],
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:

        runner = engine.Engine(
            proto_version=engine.ProtoVersion.V2,
            requestBuilder=self._request_builder,
            default_project_id=self.google_cloud_project_id)

        job = runner.run_sweep(
            program=program,
            params=params,
            repetitions=repetitions,
            processor_ids=self.processor,
            gate_set=self.gate_set)

        return job.results()
