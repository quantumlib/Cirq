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
                 engine: engine.Engine,
                 processor_id: Union[str, List[str]],
                 gate_set: 'cirq.google.SerializableGateSet'):
        """
        Args:
            engine: Quantum engine instance to use.
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
        self._processor_ids = (
            [processor_id] if isinstance(processor_id, str) else processor_id
        )
        self._gate_set = gate_set
        self._engine = engine

    def _request_builder(self, *args, **kwargs):
        from apiclient import http as apiclient_http
        request = apiclient_http.HttpRequest(*args, **kwargs)
        request.headers['X-Goog-User-Project'] = self._engine.default_project_id
        return request

    def run_sweep(
            self,
            program: Union['cirq.Circuit', 'cirq.Schedule'],
            params: 'cirq.Sweepable',
            repetitions: int = 1,
    ) -> List['cirq.TrialResult']:

        job = self._engine.run_sweep(
            program=program,
            params=params,
            repetitions=repetitions,
            processor_ids=self.processor_ids,
            gate_set=self.gate_set)

        return job.results()
