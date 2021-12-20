# Copyright 2018 The Cirq Developers
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

"""Client for running on Google's Quantum Engine.
"""

from cirq_google.engine.abstract_engine import (
    AbstractEngine,
)

from cirq_google.engine.abstract_job import (
    AbstractJob,
)

from cirq_google.engine.abstract_processor import (
    AbstractProcessor,
)

from cirq_google.engine.abstract_program import (
    AbstractProgram,
)

from cirq_google.engine.abstract_local_engine import (
    AbstractLocalEngine,
)

from cirq_google.engine.abstract_local_job import (
    AbstractLocalJob,
)

from cirq_google.engine.abstract_local_processor import (
    AbstractLocalProcessor,
)

from cirq_google.engine.abstract_local_program import (
    AbstractLocalProgram,
)

from cirq_google.engine.simulated_local_engine import (
    SimulatedLocalEngine,
)

from cirq_google.engine.simulated_local_job import (
    SimulatedLocalJob,
)

from cirq_google.engine.simulated_local_processor import (
    SimulatedLocalProcessor,
)

from cirq_google.engine.simulated_local_program import (
    SimulatedLocalProgram,
)
from cirq_google.engine.calibration import (
    Calibration,
)

from cirq_google.engine.calibration_layer import (
    CalibrationLayer,
)
from cirq_google.engine.calibration_result import (
    CalibrationResult,
)
from cirq_google.engine.engine import (
    Engine,
    get_engine,
    get_engine_calibration,
    get_engine_device,
    ProtoVersion,
)

from cirq_google.engine.engine_client import (
    EngineException,
)

from cirq_google.engine.engine_job import (
    EngineJob,
)

from cirq_google.engine.engine_processor import (
    EngineProcessor,
)

from cirq_google.engine.engine_program import (
    EngineProgram,
)

from cirq_google.engine.runtime_estimator import (
    estimate_run_time,
    estimate_run_batch_time,
    estimate_run_sweep_time,
)

from cirq_google.engine.engine_sampler import (
    get_engine_sampler,
    QuantumEngineSampler,
)

from cirq_google.engine.validating_sampler import (
    ValidatingSampler,
)
