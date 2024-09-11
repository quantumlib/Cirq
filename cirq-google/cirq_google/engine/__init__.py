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

from cirq_google.engine.abstract_engine import AbstractEngine as AbstractEngine

from cirq_google.engine.abstract_job import AbstractJob as AbstractJob

from cirq_google.engine.abstract_processor import AbstractProcessor as AbstractProcessor

from cirq_google.engine.abstract_program import AbstractProgram as AbstractProgram

from cirq_google.engine.abstract_local_engine import AbstractLocalEngine as AbstractLocalEngine

from cirq_google.engine.abstract_local_job import AbstractLocalJob as AbstractLocalJob

from cirq_google.engine.abstract_local_processor import (
    AbstractLocalProcessor as AbstractLocalProcessor,
)

from cirq_google.engine.abstract_local_program import AbstractLocalProgram as AbstractLocalProgram

from cirq_google.engine.simulated_local_engine import SimulatedLocalEngine as SimulatedLocalEngine

from cirq_google.engine.simulated_local_job import SimulatedLocalJob as SimulatedLocalJob

from cirq_google.engine.simulated_local_processor import (
    SimulatedLocalProcessor as SimulatedLocalProcessor,
)

from cirq_google.engine.simulated_local_program import (
    SimulatedLocalProgram as SimulatedLocalProgram,
)

from cirq_google.engine.calibration import Calibration as Calibration

from cirq_google.engine.calibration_layer import CalibrationLayer as CalibrationLayer

from cirq_google.engine.calibration_result import CalibrationResult as CalibrationResult

from cirq_google.engine.calibration_to_noise_properties import (
    noise_properties_from_calibration as noise_properties_from_calibration,
)

from cirq_google.engine.engine import (
    Engine as Engine,
    get_engine as get_engine,
    get_engine_calibration as get_engine_calibration,
    get_engine_device as get_engine_device,
    get_engine_sampler as get_engine_sampler,
    ProtoVersion as ProtoVersion,
)

from cirq_google.engine.engine_client import EngineException as EngineException

from cirq_google.engine.engine_job import EngineJob as EngineJob

from cirq_google.engine.engine_processor import EngineProcessor as EngineProcessor

from cirq_google.engine.engine_program import EngineProgram as EngineProgram

from cirq_google.engine.runtime_estimator import (
    estimate_run_time as estimate_run_time,
    estimate_run_batch_time as estimate_run_batch_time,
    estimate_run_sweep_time as estimate_run_sweep_time,
)

from cirq_google.engine.validating_sampler import ValidatingSampler as ValidatingSampler

from cirq_google.engine.virtual_engine_factory import (
    # pylint: disable=line-too-long
    create_default_noisy_quantum_virtual_machine as create_default_noisy_quantum_virtual_machine,
    create_device_from_processor_id as create_device_from_processor_id,
    create_noiseless_virtual_engine_from_device as create_noiseless_virtual_engine_from_device,
    create_noiseless_virtual_engine_from_proto as create_noiseless_virtual_engine_from_proto,
    create_noiseless_virtual_engine_from_templates as create_noiseless_virtual_engine_from_templates,
    create_noiseless_virtual_engine_from_latest_templates as create_noiseless_virtual_engine_from_latest_templates,
    load_median_device_calibration as load_median_device_calibration,
    load_sample_device_zphase as load_sample_device_zphase,
)

from cirq_google.engine.engine_result import EngineResult as EngineResult

from cirq_google.engine.processor_sampler import ProcessorSampler as ProcessorSampler
