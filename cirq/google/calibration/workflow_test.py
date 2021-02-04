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
from unittest import mock
import numpy as np

import cirq
import cirq.google.calibration.workflow as workflow

from cirq.google.calibration.phased_fsim import (
    FloquetPhasedFSimCalibrationOptions,
    FloquetPhasedFSimCalibrationRequest,
    PhasedFSimCharacterization,
    PhasedFSimCalibrationResult,
)


def test_run_characterization_empty():
    assert workflow.run_characterizations([], None, 'qproc', cirq.google.FSIM_GATESET) == []


@mock.patch('cirq.google.engine.Engine')
def test_run_characterization(engine):
    q_00, q_01, q_02, q_03 = [cirq.GridQubit(0, index) for index in range(4)]
    gate = cirq.FSimGate(theta=np.pi / 4, phi=0.0)
    request = FloquetPhasedFSimCalibrationRequest(
        gate=gate,
        pairs=((q_00, q_01), (q_02, q_03)),
        options=FloquetPhasedFSimCalibrationOptions(
            characterize_theta=True,
            characterize_zeta=True,
            characterize_chi=False,
            characterize_gamma=False,
            characterize_phi=True,
        ),
    )
    result = cirq.google.CalibrationResult(
        code=cirq.google.api.v2.calibration_pb2.SUCCESS,
        error_message=None,
        token=None,
        valid_until=None,
        metrics=cirq.google.Calibration(
            cirq.google.api.v2.metrics_pb2.MetricsSnapshot(
                metrics=[
                    cirq.google.api.v2.metrics_pb2.Metric(
                        name='angles',
                        targets=[
                            '0_qubit_a',
                            '0_qubit_b',
                            '0_theta_est',
                            '0_zeta_est',
                            '0_phi_est',
                            '1_qubit_a',
                            '1_qubit_b',
                            '1_theta_est',
                            '1_zeta_est',
                            '1_phi_est',
                        ],
                        values=[
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_0'),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_1'),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.1),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.2),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.3),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_2'),
                            cirq.google.api.v2.metrics_pb2.Value(str_val='0_3'),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.4),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.5),
                            cirq.google.api.v2.metrics_pb2.Value(double_val=0.6),
                        ],
                    )
                ]
            )
        ),
    )
    job = cirq.google.engine.EngineJob('', '', '', None)
    job._calibration_results = [result]
    engine.run_calibration.return_value = job
    actual = workflow.run_characterizations([request], engine, 'qproc', cirq.google.FSIM_GATESET)
    expected = [
        PhasedFSimCalibrationResult(
            parameters={
                (q_00, q_01): PhasedFSimCharacterization(
                    theta=0.1, zeta=0.2, chi=None, gamma=None, phi=0.3
                ),
                (q_02, q_03): PhasedFSimCharacterization(
                    theta=0.4, zeta=0.5, chi=None, gamma=None, phi=0.6
                ),
            },
            gate=gate,
            options=FloquetPhasedFSimCalibrationOptions(
                characterize_theta=True,
                characterize_zeta=True,
                characterize_chi=False,
                characterize_gamma=False,
                characterize_phi=True,
            ),
        )
    ]
    assert actual == expected
