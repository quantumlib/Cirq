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

from cirq.google.calibration.engine_simulator import (
    PhasedFSimEngineSimulator,
)

from cirq.google.calibration.phased_fsim import (
    ALL_ANGLES_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    IncompatibleMomentError,
    PhasedFSimCalibrationRequest,
    PhasedFSimCalibrationResult,
    PhasedFSimCharacterization,
    SQRT_ISWAP_PARAMETERS,
    SQRT_ISWAP_INV_PARAMETERS,
    THETA_ZETA_GAMMA_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    WITHOUT_CHI_FLOQUET_PHASED_FSIM_CHARACTERIZATION,
    merge_matching_results,
)

from cirq.google.calibration.workflow import (
    CircuitWithCalibration,
    FSimPhaseCorrections,
    make_zeta_chi_gamma_compensation_for_moments,
    make_zeta_chi_gamma_compensation_for_operations,
    prepare_floquet_characterization_for_moments,
    prepare_floquet_characterization_for_moment,
    prepare_floquet_characterization_for_operations,
    run_calibrations,
    run_floquet_characterization_for_moments,
    run_zeta_chi_gamma_compensation_for_moments,
    try_convert_sqrt_iswap_to_fsim,
)

# pylint: disable=wrong-import-order
from typing import Dict, Tuple

import sys as _sys
from cirq._compat import deprecate_attributes as _deprecate_attributes

deprecated_constants: Dict[str, Tuple[str, str]] = {
    'SQRT_ISWAP_PARAMETERS': ('v0.12', 'Use cirq.google.SQRT_ISWAP_INV_PARAMETERS instead'),
}
_sys.modules[__name__] = _deprecate_attributes(_sys.modules[__name__], deprecated_constants)
