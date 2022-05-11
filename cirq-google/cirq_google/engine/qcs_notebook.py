# Copyright 2021 The Cirq Developers
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

import dataclasses
from typing import Union, Optional

import cirq

from cirq_google import (
    PhasedFSimEngineSimulator,
    QuantumEngineSampler,
    Sycamore,
    SQRT_ISWAP_INV_PARAMETERS,
    PhasedFSimCharacterization,
    get_engine,
)


@dataclasses.dataclass
class QCSObjectsForNotebook:
    device: cirq.Device
    sampler: Union[PhasedFSimEngineSimulator, QuantumEngineSampler]
    signed_in: bool

    @property
    def is_simulator(self):
        return isinstance(self.sampler, PhasedFSimEngineSimulator)


# Disable missing-raises-doc lint check, since pylint gets confused
# by exceptions that are raised and caught within this function.
# pylint: disable=missing-raises-doc
def get_qcs_objects_for_notebook(
    project_id: Optional[str] = None, processor_id: Optional[str] = None
) -> QCSObjectsForNotebook:  # pragma: nocover
    """Authenticates on Google Cloud, can return a Device and Simulator.

    Args:
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).

    Returns:
        An instance of DeviceSamplerInfo.
    """

    # Check for Google Application Default Credentials and run
    # interactive login if the notebook is executed in Colab. In
    # case the notebook is executed in Jupyter notebook or other
    # IPython runtimes, no interactive login is provided, it is
    # assumed that the `GOOGLE_APPLICATION_CREDENTIALS` env var is
    # set or `gcloud auth application-default login` was executed
    # already. For more information on using Application Default Credentials
    # see https://cloud.google.com/docs/authentication/production
    try:
        from google.colab import auth
    except ImportError:
        print("Not running in a colab kernel. Will use Application Default Credentials.")
    else:
        print("Getting OAuth2 credentials.")
        print("Press enter after entering the verification code.")
        try:
            auth.authenticate_user(clear_output=False)
            print("Authentication complete.")
        except Exception as exc:
            print(f"Authentication failed: {exc}")

    # Attempt to connect to the Quantum Engine API, and use a simulator if unable to connect.
    sampler: Union[PhasedFSimEngineSimulator, QuantumEngineSampler]
    try:
        engine = get_engine(project_id)
        if processor_id:
            processor = engine.get_processor(processor_id)
        else:
            processors = engine.list_processors()
            if not processors:
                raise ValueError("No processors available.")
            processor = processors[0]
            print(f"Available processors: {[p.processor_id for p in processors]}")
            print(f"Using processor: {processor.processor_id}")
        device = processor.get_device()
        sampler = processor.get_sampler()
        signed_in = True
    except Exception as exc:
        print(f"Unable to connect to quantum engine: {exc}")
        print("Using a noisy simulator.")
        sampler = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
            mean=SQRT_ISWAP_INV_PARAMETERS,
            sigma=PhasedFSimCharacterization(theta=0.01, zeta=0.10, chi=0.01, gamma=0.10, phi=0.02),
        )
        device = Sycamore
        signed_in = False

    return QCSObjectsForNotebook(device=device, sampler=sampler, signed_in=signed_in)


# pylint: enable=missing-raises-doc
