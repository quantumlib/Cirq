# Copyright 2022 The Cirq Developers
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
#
from typing import Any, Dict


import cirq


class PasqalGateset(cirq.neutral_atoms.NeutralAtomGateset):
    """A Compilation target intended for Pasqal neutral atom devices.
    This gateset supports single qubit gates that can be used
    in a parallel fashion as well as CZ.

    This gateset can optionally include CNOT, CCNOT (TOFFOLI) gates, and
    CCZ as well.

    Args:
        include_additional_controlled_ops: Whether to include CCZ, CCNOT, and CNOT
            gates (defaults to True).
    """

    def __init__(self, include_additional_controlled_ops: bool = True):
        gate_families = [
            cirq.ParallelGateFamily(cirq.H),
            cirq.ParallelGateFamily(cirq.PhasedXPowGate),
            cirq.ParallelGateFamily(cirq.XPowGate),
            cirq.ParallelGateFamily(cirq.YPowGate),
            cirq.ParallelGateFamily(cirq.ZPowGate),
            cirq.AnyIntegerPowerGateFamily(cirq.CZPowGate),
            cirq.IdentityGate,
            cirq.MeasurementGate,
        ]
        self.include_additional_controlled_ops = include_additional_controlled_ops
        if self.include_additional_controlled_ops:
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CNotPowGate))
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CCNotPowGate))
            gate_families.append(cirq.AnyIntegerPowerGateFamily(cirq.CCZPowGate))

        # Call cirq.Gateset __init__ which is our grand-father inherited class
        # pylint doesn't like this so disable checks on this.
        # pylint: disable=bad-super-call
        super(cirq.neutral_atoms.NeutralAtomGateset, self).__init__(
            *gate_families, unroll_circuit_op=False
        )

    def __repr__(self):
        return (
            f'cirq_pasqal.PasqalGateset(include_additional_controlled_ops='
            f'{self.include_additional_controlled_ops})'
        )

    @classmethod
    def _from_json_dict_(cls, include_additional_controlled_ops, **kwargs):
        return cls(include_additional_controlled_ops=include_additional_controlled_ops)

    def _json_dict_(self) -> Dict[str, Any]:
        return cirq.protocols.obj_to_dict_helper(self, ['include_additional_controlled_ops'])
