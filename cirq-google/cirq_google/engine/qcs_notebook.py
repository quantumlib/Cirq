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

import os
import dataclasses
from typing import Union, Optional

import cirq

from cirq_google import (
    PhasedFSimEngineSimulator,
    QuantumEngineSampler,
    Sycamore,
    SQRT_ISWAP_INV_PARAMETERS,
    PhasedFSimCharacterization,
    get_engine_sampler,
    get_engine_device,
)


@dataclasses.dataclass
class QCSObjectsForNotebook:
    device: cirq.Device
    sampler: Union[PhasedFSimEngineSimulator, QuantumEngineSampler]
    signed_in: bool

    @property
    def is_simulator(self):
        return isinstance(self.sampler, PhasedFSimEngineSimulator)


def get_qcs_objects_for_notebook(
    project_id: Optional[str] = None, processor_id: Optional[str] = None
) -> QCSObjectsForNotebook:
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

        # Following code runs the user through the Colab OAuth process.

        # Checks for Google Application Default Credentials and runs
        # interactive login if the notebook is executed in Colab. In
        # case the notebook is executed in Jupyter notebook or other
        # IPython runtimes, no interactive login is provided, it is
        # assumed that the `GOOGLE_APPLICATION_CREDENTIALS` env var is
        # set or `gcloud auth application-default login` was executed
        # already. For more information on using Application Default Credentials
        # see https://cloud.google.com/docs/authentication/production

        in_colab = False
        try:
            from IPython import get_ipython

            in_colab = 'google.colab' in str(get_ipython())

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
        except:
            pass

        # End of Google Colab Authentication segment

    device: cirq.Device
    sampler: Union[PhasedFSimEngineSimulator, QuantumEngineSampler]
    if google_cloud_signin_failed or processor_id is None:
        print("Using a noisy simulator.")
        sampler = PhasedFSimEngineSimulator.create_with_random_gaussian_sqrt_iswap(
            mean=SQRT_ISWAP_INV_PARAMETERS,
            sigma=PhasedFSimCharacterization(theta=0.01, zeta=0.10, chi=0.01, gamma=0.10, phi=0.02),
        )
        device = Sycamore
    else:  # pragma: no cover
        device = get_engine_device(processor_id)
        sampler = get_engine_sampler(processor_id, gate_set_name="sqrt_iswap")
    return QCSObjectsForNotebook(
        device=device,
        sampler=sampler,
        signed_in=not google_cloud_signin_failed,
    )
