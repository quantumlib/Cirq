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
from typing import cast, Optional, Sequence, Union

import cirq

from cirq_google import ProcessorSampler, get_engine
from cirq_google.engine import (
    AbstractEngine,
    AbstractProcessor,
    AbstractLocalProcessor,
    create_noiseless_virtual_engine_from_latest_templates,
    EngineProcessor,
)


@dataclasses.dataclass
class QCSObjectsForNotebook:
    """All the objects you might need to run a notbook with QCS.

    Contains an (Abstract) Engine, Processor, Device, and Sampler,
    as well as associated meta-data signed_in, processor_id, and project_id.

    This removes the need for boiler plate in notebooks, and provides a
    central place to handle the various environments (testing vs production),
    (stand-alone vs colab vs jupyter).
    """

    engine: AbstractEngine
    processor: AbstractProcessor
    device: cirq.Device
    sampler: ProcessorSampler
    signed_in: bool
    processor_id: Optional[str]
    project_id: Optional[str]
    is_simulator: bool


def get_qcs_objects_for_notebook(
    project_id: Optional[str] = None, processor_id: Optional[str] = None, virtual=False
) -> QCSObjectsForNotebook:
    """Authenticates on Google Cloud and returns Engine related objects.

    This function will authenticate to Google Cloud and attempt to
    instantiate an Engine object.  If it does not succeed, it will instead
    return a virtual AbstractEngine that is backed by a noisy simulator.
    This function is designed for maximum versatility and
    to work in colab notebooks, as a stand-alone, and in tests.

    Note that, if you are using this to connect to QCS and do not care about
    the added versatility, you may want to use `cirq_google.get_engine()` or
    `cirq_google.Engine()` instead to guarantee the use of a production instance
    and to avoid accidental use of a noisy simulator.

    Args:
        project_id: Optional explicit Google Cloud project id. Otherwise,
            this defaults to the environment variable GOOGLE_CLOUD_PROJECT.
            By using an environment variable, you can avoid hard-coding
            personal project IDs in shared code.
        processor_id: Engine processor ID (from Cloud console or
            ``Engine.list_processors``).
        virtual: If set to True, will create a noisy virtual Engine instead.
            This is useful for testing and simulation.

    Returns:
        An instance of QCSObjectsForNotebook which contains all the objects .

    Raises:
        ValueError: if processor_id is not specified and no processors are available.
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
    if virtual:
        engine: AbstractEngine = create_noiseless_virtual_engine_from_latest_templates()
        signed_in = False
        is_simulator = True
    else:
        try:
            engine = get_engine(project_id)
            signed_in = True
            is_simulator = False
        except Exception as exc:
            print(f"Unable to connect to quantum engine: {exc}")
            print("Using a noisy simulator.")
            engine = create_noiseless_virtual_engine_from_latest_templates()
            signed_in = False
            is_simulator = True
    if processor_id:
        processor = engine.get_processor(processor_id)
    else:
        # All of these are either local processors or engine processors
        # Either way, tell mypy they have a processor_id field.
        processors = cast(
            Sequence[Union[EngineProcessor, AbstractLocalProcessor]], engine.list_processors()
        )
        if not processors:
            raise ValueError("No processors available.")
        processor = processors[0]
        processor_id = processor.processor_id
        print(f"Available processors: {[p.processor_id for p in processors]}")
        print(f"Using processor: {processor_id}")
    if not project_id:
        project_id = getattr(processor, 'project_id', None)
    device = processor.get_device()
    sampler = processor.get_sampler()
    return QCSObjectsForNotebook(
        engine=engine,
        processor=processor,
        device=device,
        sampler=sampler,
        signed_in=signed_in,
        project_id=project_id,
        processor_id=processor_id,
        is_simulator=is_simulator,
    )
