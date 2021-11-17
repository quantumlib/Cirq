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

import multiprocessing.pool

import numpy as np
import pandas as pd
import pytest
import scipy.optimize
import scipy.optimize._minimize

import cirq
import cirq_google as cg
from cirq.experiments import random_rotations_between_grid_interaction_layers_circuit
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.calibration.phased_fsim import (
    LocalXEBPhasedFSimCalibrationOptions,
    LocalXEBPhasedFSimCalibrationRequest,
)
from cirq_google.calibration.xeb_wrapper import (
    run_local_xeb_calibration,
    _maybe_multiprocessing_pool,
)


def _minimize_patch(
    fun,
    x0,
    args=(),
    method=None,
    jac=None,
    hess=None,
    hessp=None,
    bounds=None,
    constraints=(),
    tol=None,
    callback=None,
    options=None,
    x0_should_be=None,
):
    assert method == 'nelder-mead'
    np.testing.assert_allclose(x0_should_be, x0)

    return scipy.optimize.OptimizeResult(
        fun=0,
        nit=0,
        nfev=0,
        status=0,
        success=True,
        message='monkeypatched',
        x=x0.copy(),
        final_simplex=None,
    )


def _benchmark_patch(*args, **kwargs):
    return pd.DataFrame()


@pytest.mark.parametrize(
    ['fsim_options', 'x0_should_be'],
    [
        (
            XEBPhasedFSimCharacterizationOptions(
                characterize_zeta=True,
                characterize_gamma=True,
                characterize_chi=True,
                characterize_theta=False,
                characterize_phi=False,
            ),
            [0.0, 0.0, 0.0],
        ),
        (XEBPhasedFSimCharacterizationOptions(), [np.pi / 4, 0.0, 0.0, 0.0, 0.0]),
        (
            XEBPhasedFSimCharacterizationOptions(
                characterize_zeta=True,
                characterize_chi=True,
                characterize_gamma=True,
                characterize_theta=False,
                characterize_phi=False,
                theta_default=99,
                zeta_default=0.1,
                chi_default=0.2,
                gamma_default=0.3,
                phi_default=99,
            ),
            [0.1, 0.2, 0.3],
        ),
    ],
)
def test_run_calibration(monkeypatch, fsim_options, x0_should_be):
    def _minimize_patch_2(*args, **kwargs):
        return _minimize_patch(*args, **kwargs, x0_should_be=x0_should_be)

    monkeypatch.setattr('scipy.optimize.minimize', _minimize_patch_2)
    monkeypatch.setattr(
        'cirq_google.calibration.xeb_wrapper.xebf.benchmark_2q_xeb_fidelities', _benchmark_patch
    )
    qubit_indices = [
        (0, 5),
        (0, 6),
        (1, 6),
        (2, 6),
    ]
    qubits = [cirq.GridQubit(*idx) for idx in qubit_indices]
    sampler = cirq.ZerosSampler()

    circuits = [
        random_rotations_between_grid_interaction_layers_circuit(
            qubits,
            depth=depth,
            two_qubit_op_factory=lambda a, b, _: (cirq.SQRT_ISWAP_INV.on(a, b)),
            pattern=cirq.experiments.GRID_ALIGNED_PATTERN,
            seed=10,
        )
        for depth in [5, 10]
    ]

    options = LocalXEBPhasedFSimCalibrationOptions(
        fsim_options=fsim_options,
        n_processes=1,
    )

    characterization_requests = []
    for circuit in circuits:
        _, characterization_requests = cg.prepare_characterization_for_moments(
            circuit, options=options, initial=characterization_requests
        )
    assert len(characterization_requests) == 2
    for cr in characterization_requests:
        assert isinstance(cr, LocalXEBPhasedFSimCalibrationRequest)

    characterizations = [
        run_local_xeb_calibration(request, sampler) for request in characterization_requests
    ]

    final_params = dict()
    for c in characterizations:
        final_params.update(c.parameters)
    assert len(final_params) == 3  # pairs


def test_maybe_pool():
    with _maybe_multiprocessing_pool(1) as pool:
        assert pool is None

    with _maybe_multiprocessing_pool(2) as pool:
        assert isinstance(pool, multiprocessing.pool.Pool)
