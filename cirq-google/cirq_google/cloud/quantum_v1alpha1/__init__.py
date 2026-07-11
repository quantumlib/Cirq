# -*- coding: utf-8 -*-
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import google.api_core as api_core
import sys


from importlib import metadata


from .services.quantum_engine_service import QuantumEngineServiceClient
from .services.quantum_engine_service import QuantumEngineServiceAsyncClient

from .types.engine import CancelQuantumJobRequest
from .types.engine import CancelQuantumReservationRequest
from .types.engine import CompileQecProgramRequest
from .types.engine import CompileQecProgramResponse
from .types.engine import CreateQuantumJobRequest
from .types.engine import CreateQuantumProgramAndJobRequest
from .types.engine import CreateQuantumProgramRequest
from .types.engine import CreateQuantumReservationRequest
from .types.engine import DeleteQuantumJobRequest
from .types.engine import DeleteQuantumProgramRequest
from .types.engine import DeleteQuantumReservationRequest
from .types.engine import GetQuantumCalibrationRequest
from .types.engine import GetQuantumJobRequest
from .types.engine import GetQuantumProcessorConfigRequest
from .types.engine import GetQuantumProcessorRequest
from .types.engine import GetQuantumProgramRequest
from .types.engine import GetQuantumReservationRequest
from .types.engine import GetQuantumResultRequest
from .types.engine import ListQuantumCalibrationsRequest
from .types.engine import ListQuantumCalibrationsResponse
from .types.engine import ListQuantumJobEventsRequest
from .types.engine import ListQuantumJobEventsResponse
from .types.engine import ListQuantumJobsRequest
from .types.engine import ListQuantumJobsResponse
from .types.engine import ListQuantumProcessorAutomationRunHistoryRequest
from .types.engine import ListQuantumProcessorAutomationRunHistoryResponse
from .types.engine import ListQuantumProcessorConfigsRequest
from .types.engine import ListQuantumProcessorConfigsResponse
from .types.engine import ListQuantumProcessorsRequest
from .types.engine import ListQuantumProcessorsResponse
from .types.engine import ListQuantumProgramsRequest
from .types.engine import ListQuantumProgramsResponse
from .types.engine import ListQuantumReservationBudgetsRequest
from .types.engine import ListQuantumReservationBudgetsResponse
from .types.engine import ListQuantumReservationGrantsRequest
from .types.engine import ListQuantumReservationGrantsResponse
from .types.engine import ListQuantumReservationsRequest
from .types.engine import ListQuantumReservationsResponse
from .types.engine import ListQuantumTimeSlotsRequest
from .types.engine import ListQuantumTimeSlotsResponse
from .types.engine import QuantumRunStreamRequest
from .types.engine import QuantumRunStreamResponse
from .types.engine import ReallocateQuantumReservationGrantRequest
from .types.engine import StreamError
from .types.engine import UpdateQuantumJobRequest
from .types.engine import UpdateQuantumProgramRequest
from .types.engine import UpdateQuantumReservationRequest
from .types.quantum import DeviceConfigKey
from .types.quantum import DeviceConfigSelector
from .types.quantum import ExecutionStatus
from .types.quantum import GcsLocation
from .types.quantum import InlineData
from .types.quantum import OutputConfig
from .types.quantum import QecRecipe
from .types.quantum import QuantumCalibration
from .types.quantum import QuantumJob
from .types.quantum import QuantumJobEvent
from .types.quantum import QuantumProcessor
from .types.quantum import QuantumProcessorAutomationRunHistory
from .types.quantum import QuantumProcessorConfig
from .types.quantum import QuantumProgram
from .types.quantum import QuantumReservation
from .types.quantum import QuantumReservationBudget
from .types.quantum import QuantumReservationGrant
from .types.quantum import QuantumResult
from .types.quantum import QuantumTimeSlot
from .types.quantum import SchedulingConfig

if hasattr(api_core, "check_python_version") and hasattr(
    api_core, "check_dependency_versions"
):  # pragma: NO COVER
    api_core.check_python_version("cirq_google.cloud.quantum_v1alpha1")  # type: ignore
    api_core.check_dependency_versions("cirq_google.cloud.quantum_v1alpha1")  # type: ignore
else:  # pragma: NO COVER
    # An older version of api_core is installed which does not define the
    # functions above. We do equivalent checks manually.
    try:
        import warnings

        _py_version_str = sys.version.split()[0]
        _package_label = "cirq_google.cloud.quantum_v1alpha1"
        if sys.version_info < (3, 10):
            warnings.warn(
                "You are using a non-supported Python version "
                + f"({_py_version_str}).  Google will not post any further "
                + f"updates to {_package_label} supporting this Python version. "
                + "Please upgrade to the latest Python version, or at "
                + f"least to Python 3.10, and then update {_package_label}.",
                FutureWarning,
            )

        def parse_version_to_tuple(version_string: str):
            """Safely converts a semantic version string to a comparable tuple of integers.
            Example: "6.33.5" -> (6, 33, 5)
            Ignores non-numeric parts and handles common version formats.
            Args:
                version_string: Version string in the format "x.y.z" or "x.y.z<suffix>"
            Returns:
                Tuple of integers for the parsed version string.
            """
            parts = []
            for part in version_string.split("."):
                try:
                    parts.append(int(part))
                except ValueError:
                    # If it's a non-numeric part (e.g., '1.0.0b1' -> 'b1'), stop here.
                    # This is a simplification compared to 'packaging.parse_version', but sufficient
                    # for comparing strictly numeric semantic versions.
                    break
            return tuple(parts)

        def _get_version(dependency_name):
            try:
                version_string: str = metadata.version(dependency_name)
                parsed_version = parse_version_to_tuple(version_string)
                return (parsed_version, version_string)
            except Exception:
                # Catch exceptions from metadata.version() (e.g., PackageNotFoundError)
                # or errors during parse_version_to_tuple
                return (None, "--")

        _dependency_package = "google.protobuf"
        _next_supported_version = "6.33.5"
        _next_supported_version_tuple = (6, 33, 5)
        _recommendation = " (we recommend 7.x)"
        _version_used, _version_used_string = _get_version(_dependency_package)
        if _version_used and _version_used < _next_supported_version_tuple:
            warnings.warn(
                f"Package {_package_label} depends on "
                + f"{_dependency_package}, currently installed at version "
                + f"{_version_used_string}. Future updates to "
                + f"{_package_label} will require {_dependency_package} at "
                + f"version {_next_supported_version} or higher{_recommendation}."
                + " Please ensure "
                + "that either (a) your Python environment doesn't pin the "
                + f"version of {_dependency_package}, so that updates to "
                + f"{_package_label} can require the higher version, or "
                + "(b) you manually update your Python environment to use at "
                + f"least version {_next_supported_version} of "
                + f"{_dependency_package}.",
                FutureWarning,
            )
    except Exception:
        warnings.warn(
            "Could not determine the version of Python "
            + "currently being used. To continue receiving "
            + "updates for {_package_label}, ensure you are "
            + "using a supported version of Python; see "
            + "https://devguide.python.org/versions/"
        )

__all__ = (
    'QuantumEngineServiceAsyncClient',
    'CancelQuantumJobRequest',
    'CancelQuantumReservationRequest',
    'CompileQecProgramRequest',
    'CompileQecProgramResponse',
    'CreateQuantumJobRequest',
    'CreateQuantumProgramAndJobRequest',
    'CreateQuantumProgramRequest',
    'CreateQuantumReservationRequest',
    'DeleteQuantumJobRequest',
    'DeleteQuantumProgramRequest',
    'DeleteQuantumReservationRequest',
    'DeviceConfigKey',
    'DeviceConfigSelector',
    'ExecutionStatus',
    'GcsLocation',
    'GetQuantumCalibrationRequest',
    'GetQuantumJobRequest',
    'GetQuantumProcessorConfigRequest',
    'GetQuantumProcessorRequest',
    'GetQuantumProgramRequest',
    'GetQuantumReservationRequest',
    'GetQuantumResultRequest',
    'InlineData',
    'ListQuantumCalibrationsRequest',
    'ListQuantumCalibrationsResponse',
    'ListQuantumJobEventsRequest',
    'ListQuantumJobEventsResponse',
    'ListQuantumJobsRequest',
    'ListQuantumJobsResponse',
    'ListQuantumProcessorAutomationRunHistoryRequest',
    'ListQuantumProcessorAutomationRunHistoryResponse',
    'ListQuantumProcessorConfigsRequest',
    'ListQuantumProcessorConfigsResponse',
    'ListQuantumProcessorsRequest',
    'ListQuantumProcessorsResponse',
    'ListQuantumProgramsRequest',
    'ListQuantumProgramsResponse',
    'ListQuantumReservationBudgetsRequest',
    'ListQuantumReservationBudgetsResponse',
    'ListQuantumReservationGrantsRequest',
    'ListQuantumReservationGrantsResponse',
    'ListQuantumReservationsRequest',
    'ListQuantumReservationsResponse',
    'ListQuantumTimeSlotsRequest',
    'ListQuantumTimeSlotsResponse',
    'OutputConfig',
    'QecRecipe',
    'QuantumCalibration',
    'QuantumEngineServiceClient',
    'QuantumJob',
    'QuantumJobEvent',
    'QuantumProcessor',
    'QuantumProcessorAutomationRunHistory',
    'QuantumProcessorConfig',
    'QuantumProgram',
    'QuantumReservation',
    'QuantumReservationBudget',
    'QuantumReservationGrant',
    'QuantumResult',
    'QuantumRunStreamRequest',
    'QuantumRunStreamResponse',
    'QuantumTimeSlot',
    'ReallocateQuantumReservationGrantRequest',
    'SchedulingConfig',
    'StreamError',
    'UpdateQuantumJobRequest',
    'UpdateQuantumProgramRequest',
    'UpdateQuantumReservationRequest',
)
