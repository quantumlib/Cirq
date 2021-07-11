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
from typing import List, TYPE_CHECKING, Union, Optional, cast, Tuple

import cirq
from cirq_google import engine, gate_sets

if TYPE_CHECKING:
    import cirq_google


class QuantumEngineSampler(cirq.Sampler):
    """A sampler that samples from processors managed by the Quantum Engine.

    Exposes a `cirq_google.Engine` instance as a `cirq.Sampler`.
    """

    def __init__(
        self,
        *,
        engine: 'cirq_google.Engine',
        processor_id: Union[str, List[str]],
        gate_set: 'cirq_google.SerializableGateSet',
    ):
        """
        Args:
            engine: Quantum engine instance to use.
            processor_id: String identifier, or list of string identifiers,
                determining which processors may be used when sampling.
            gate_set: Determines how to serialize circuits when requesting
                samples.
        """
        self._processor_ids = [processor_id] if isinstance(processor_id, str) else processor_id
        self._gate_set = gate_set
        self._engine = engine

    def run_sweep(
        self,
        program: Union[cirq.Circuit, 'cirq_google.EngineProgram'],
        params: cirq.Sweepable,
        repetitions: int = 1,
    ) -> List[cirq.Result]:
        if isinstance(program, engine.EngineProgram):
            job = program.run_sweep(
                params=params, repetitions=repetitions, processor_ids=self._processor_ids
            )
        else:
            job = self._engine.run_sweep(
                program=cast(cirq.Circuit, program),
                params=params,
                repetitions=repetitions,
                processor_ids=self._processor_ids,
                gate_set=self._gate_set,
            )
        return job.results()

    def run_batch(
        self,
        programs: List[cirq.Circuit],
        params_list: Optional[List[cirq.Sweepable]] = None,
        repetitions: Union[int, List[int]] = 1,
    ) -> List[List[cirq.Result]]:
        """Runs the supplied circuits.

        In order to gain a speedup from using this method instead of other run
        methods, the following conditions must be satisfied:
            1. All circuits must measure the same set of qubits.
            2. The number of circuit repetitions must be the same for all
               circuits. That is, the `repetitions` argument must be an integer,
               or else a list with identical values.
        """
        if isinstance(repetitions, List) and len(programs) != len(repetitions):
            raise ValueError(
                'len(programs) and len(repetitions) must match. '
                f'Got {len(programs)} and {len(repetitions)}.'
            )
        if isinstance(repetitions, int) or len(set(repetitions)) == 1:
            # All repetitions are the same so batching can be done efficiently
            if isinstance(repetitions, List):
                repetitions = repetitions[0]
            job = self._engine.run_batch(
                programs=programs,
                params_list=params_list,
                repetitions=repetitions,
                processor_ids=self._processor_ids,
                gate_set=self._gate_set,
            )
            return job.batched_results()
        # Varying number of repetitions so no speedup
        return super().run_batch(programs, params_list, repetitions)

    @property
    def engine(self) -> 'cirq_google.Engine':
        return self._engine


def get_engine_sampler(
    processor_id: str, gate_set_name: str, project_id: Optional[str] = None
) -> 'cirq_google.QuantumEngineSampler':
    """Get an EngineSampler assuming some sensible defaults.

    This uses the environment variable GOOGLE_CLOUD_PROJECT for the Engine
    project_id, unless set explicitly.

    Args:
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        gate_set_name: One of ['sqrt_iswap', 'sycamore'].
            See `cirq_google.NAMED_GATESETS`.
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.

    Returns:
        A `QuantumEngineSampler` instance.

    Raises:
         ValueError: If the supplied gate set is not a supported gate set name.
         EnvironmentError: If no project_id is specified and the environment
            variable GOOGLE_CLOUD_PROJECT is not set.
    """
    if gate_set_name not in gate_sets.NAMED_GATESETS:
        raise ValueError(
            f"Unknown gateset {gate_set_name}. Please use one of: "
            f"{sorted(gate_sets.NAMED_GATESETS.keys())}."
        )
    gate_set = gate_sets.NAMED_GATESETS[gate_set_name]
    return engine.get_engine(project_id).sampler(processor_id=processor_id, gate_set=gate_set)


def get_device_sampler(
    project_id: Optional[str] = None, processor_id: Optional[str] = None, get_simulator: bool = True
) -> Tuple[
    Tuple[Union[cirq.Device], int],
    Union['cirq_google.PhasedFSimEngineSimulator', 'cirq_google.QuantumEngineSampler'],
    bool,
]:
    """Authenticates on Google Cloud, can return a Device and Simulator.

    This uses the environment variable GOOGLE_CLOUD_PROJECT for the Engine
    project_id, unless set explicitly.

    Args:
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).

    Returns:
        A tuple of ((`Device`, `int`), `Simulator/Sampler`, `bool`). The first element is the
        device and it's corresponding line length, the second is a simulator instance, and the
        third is a boolean value, true if the signin was successful, false otherwise.
    """
    import os
    from cirq_google import (
        PhasedFSimEngineSimulator,
        SQRT_ISWAP_INV_PARAMETERS,
        PhasedFSimCharacterization,
        Bristlecone,
        get_engine_device,
        get_engine_sampler,
    )

    # Converting empty strings to None for form field inputs
    if project_id == "":
        project_id = None
    if processor_id == "":
        processor_id = None

    google_cloud_signin_failed: bool = False
    if project_id is None:
        if 'GOOGLE_CLOUD_PROJECT' not in os.environ:
            print("No project_id provided and environment variable GOOGLE_CLOUD_PROJECT not set.")
            google_cloud_signin_failed = True
    else:  # pragma: no cover
        os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

        def authenticate_user():
            """Runs the user through the Colab OAuth process.

            Checks for Google Application Default Credentials and runs
            interactive login if the notebook is executed in Colab. In
            case the notebook is executed in Jupyter notebook or other
            IPython runtimes, no interactive login is provided, it is
            assumed that the `GOOGLE_APPLICATION_CREDENTIALS` env var is
            set or `gcloud auth application-default login` was executed
            already.

            For more information on using Application Default Credentials see
            https://cloud.google.com/docs/authentication/production
            """
            in_colab = False
            try:
                from IPython import get_ipython

                in_colab = 'google.colab' in str(get_ipython())
            except:
                return

            if in_colab:
                from google.colab import auth

                print("Getting OAuth2 credentials.")
                print("Press enter after entering the verification code.")
                auth.authenticate_user(clear_output=False)
                print("Authentication complete.")
            else:
                print(
                    "Notebook isn't executed with Colab, assuming "
                    "Application Default Credentials are setup."
                )

        authenticate_user()

    device: cirq.Device
    sampler: Union['cirq_google.PhasedFSimEngineSimulator', 'cirq_google.QuantumEngineSampler']
    if google_cloud_signin_failed or processor_id is None:
        print("Using a noisy simulator.")
        sampler = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
            mean=SQRT_ISWAP_INV_PARAMETERS,
            sigma=PhasedFSimCharacterization(theta=0.01, zeta=0.10, chi=0.01, gamma=0.10, phi=0.02),
        )
        device = Bristlecone
        line_length = 20
    else:  # pragma: no cover
        device = get_engine_device(processor_id)
        sampler = get_engine_sampler(processor_id, gate_set_name="sqrt_iswap")
        line_length = 35
    return (device, line_length), sampler, not google_cloud_signin_failed
