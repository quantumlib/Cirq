# Copyright 2024 The Cirq Developers
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

import unittest.mock
import pytest
import numpy as np
import cirq
from cirq.transformers.gauge_compiling import (
    GaugeTransformer,
    CZGaugeTransformer,
    ConstantGauge,
    GaugeSelector,
)
from cirq.transformers.analytical_decompositions import single_qubit_decompositions


def test_deep_transformation_not_supported():

    with pytest.raises(ValueError, match="cannot be used with deep=True"):
        _ = GaugeTransformer(target=cirq.CZ, gauge_selector=lambda _: None)(
            cirq.Circuit(), context=cirq.TransformerContext(deep=True)
        )

    with pytest.raises(ValueError, match="cannot be used with deep=True"):
        _ = GaugeTransformer(target=cirq.CZ, gauge_selector=lambda _: None).as_sweep(
            cirq.Circuit(), context=cirq.TransformerContext(deep=True), N=1
        )


def test_ignore_tags():
    c = cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)).with_tags('foo'))
    assert c == CZGaugeTransformer(c, context=cirq.TransformerContext(tags_to_ignore={"foo"}))
    parameterized_circuit, _ = CZGaugeTransformer.as_sweep(
        c, context=cirq.TransformerContext(tags_to_ignore={"foo"}), N=1
    )
    assert c == parameterized_circuit


def test_target_can_be_gateset():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(*qs))
    transformer = GaugeTransformer(
        target=cirq.Gateset(cirq.CZ), gauge_selector=CZGaugeTransformer.gauge_selector
    )
    want = cirq.Circuit(cirq.Y.on_each(qs), cirq.CZ(*qs), cirq.X.on_each(qs))
    assert transformer(c, prng=np.random.default_rng(0)) == want


def test_as_sweep_multi_pre_or_multi_post():
    transformer = GaugeTransformer(
        target=cirq.CZ,
        gauge_selector=GaugeSelector(
            gauges=[
                ConstantGauge(
                    two_qubit_gate=cirq.CZ,
                    support_sweep=True,
                    pre_q0=[cirq.X, cirq.X],
                    post_q0=[cirq.Z],
                    pre_q1=[cirq.Y],
                    post_q1=[cirq.Y, cirq.Y, cirq.Y],
                )
            ]
        ),
    )
    qs = cirq.LineQubit.range(2)
    input_circuit = cirq.Circuit(cirq.CZ(*qs))
    parameterized_circuit, sweeps = transformer.as_sweep(input_circuit, N=1)

    for params in sweeps:
        compiled_circuit = cirq.resolve_parameters(parameterized_circuit, params)
        cirq.testing.assert_circuits_have_same_unitary_given_final_permutation(
            input_circuit, compiled_circuit, qubit_map={q: q for q in input_circuit.all_qubits()}
        )


def test_as_sweep_invalid_gauge_sequence():
    transfomer = GaugeTransformer(
        target=cirq.CZ,
        gauge_selector=GaugeSelector(
            gauges=[
                ConstantGauge(
                    two_qubit_gate=cirq.CZ,
                    support_sweep=True,
                    pre_q0=[cirq.measure],
                    post_q0=[cirq.Z],
                    pre_q1=[cirq.X],
                    post_q1=[cirq.Z],
                )
            ]
        ),
    )
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(*qs))
    with pytest.raises(ValueError, match="Invalid gate sequence to be converted to PhasedXZGate."):
        transfomer.as_sweep(c, N=1)


def test_as_sweep_convert_to_phxz_failed():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(*qs))

    def mock_single_qubit_matrix_to_phxz(*args, **kwargs):
        # Return an non PhasedXZ gate, so we expect errors from as_sweep().
        return cirq.X

    with unittest.mock.patch.object(
        single_qubit_decompositions,
        "single_qubit_matrix_to_phxz",
        new=mock_single_qubit_matrix_to_phxz,
    ):
        with pytest.raises(
            ValueError, match="Failed to convert the gate sequence to a PhasedXZ gate."
        ):
            _ = CZGaugeTransformer.as_sweep(c, context=cirq.TransformerContext(), N=1)
