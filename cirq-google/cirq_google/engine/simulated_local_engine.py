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
"""Classes for running against Google's Quantum Cloud Service.

As an example, to run a circuit against the xmon simulator on the cloud,
    engine = cirq_google.Engine(project_id='my-project-id')
    program = engine.create_program(circuit)
    result0 = program.run(params=params0, repetitions=10)
    result1 = program.run(params=params1, repetitions=10)

In order to run on must have access to the Quantum Engine API. Access to this
API is (as of June 22, 2018) restricted to invitation only.
"""
from typing import List

from cirq_google.engine.abstract_local_engine import AbstractLocalEngine
from cirq_google.engine.abstract_local_processor import AbstractLocalProcessor


class SimulatedLocalEngine(AbstractLocalEngine):
    """Collection of processors backed by local samplers.

    This class is a wrapper around `AbstractLocalEngine` and
    adds no additional functionality and exists for naming consistency
    and for possible future extension.

    This class assumes that all processors are local.  Processors
    are given during initialization.  Program and job querying
    functionality is done by serially querying all child processors.

    """

    def __init__(self, processors: List[AbstractLocalProcessor]):
        super().__init__(processors)
