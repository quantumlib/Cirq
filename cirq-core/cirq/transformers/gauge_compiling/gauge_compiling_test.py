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

import pytest
import numpy as np
import cirq
from cirq.transformers.gauge_compiling import GaugeTransformer, CZGaugeTransformer


def test_deep_transformation_not_supported():

    with pytest.raises(ValueError, match="cannot be used with deep=True"):
        _ = GaugeTransformer(target=cirq.CZ, gauge_selector=lambda _: None)(
            cirq.Circuit(), context=cirq.TransformerContext(deep=True)
        )


def test_ignore_tags():
    c = cirq.Circuit(cirq.CZ(*cirq.LineQubit.range(2)).with_tags('foo'))
    assert c == CZGaugeTransformer(c, context=cirq.TransformerContext(tags_to_ignore={"foo"}))


def test_target_can_be_gateset():
    qs = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.CZ(*qs))
    transformer = GaugeTransformer(
        target=cirq.Gateset(cirq.CZ), gauge_selector=CZGaugeTransformer.gauge_selector
    )
    want = cirq.Circuit(cirq.Y.on_each(qs), cirq.CZ(*qs), cirq.X.on_each(qs))
    assert transformer(c, prng=np.random.default_rng(0)) == want
