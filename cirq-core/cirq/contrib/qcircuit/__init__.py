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

"""Converts cirq circuits into latex using qcircuit."""

from cirq.contrib.qcircuit.qcircuit_diagram import (
    circuit_to_latex_using_qcircuit as circuit_to_latex_using_qcircuit,
)

from cirq.contrib.qcircuit.qcircuit_diagram_info import (
    escape_text_for_latex as escape_text_for_latex,
    get_multigate_parameters as get_multigate_parameters,
    get_qcircuit_diagram_info as get_qcircuit_diagram_info,
)
