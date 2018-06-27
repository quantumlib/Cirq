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

from typing import Dict, Callable, TypeVar

import math


TOutcome = TypeVar('TOutcome')
THypothesis = TypeVar('THypothesis')
TExperiment = TypeVar('TExperiment')


def log_loss(probability: float, repeats: int = 1) -> float:
    if repeats == 0:
        return 0
    if probability <= 0:
        return -float('inf')
    return math.log2(probability) * repeats


def binomial_log_loss(zeros: int,
                      ones: int,
                      p_one: float) -> float:
    p_zero = 1 - p_one
    return log_loss(p_zero, zeros) + log_loss(p_one, ones)


def infer_posterior(
        predict_log_loss: Callable[[THypothesis, TExperiment, TOutcome], float],
        data: Dict[TExperiment, TOutcome],
        prior: Dict[THypothesis, float]) -> Dict[THypothesis, float]:

    log_losses = {}
    for hypothesis in prior.keys():
        loss = 0
        for experiment, outcome in data.items():
            loss += predict_log_loss(hypothesis, experiment, outcome)
        log_losses[hypothesis] = loss

    # The absolute losses may be smaller than a float can represent. To avoid
    # sending everything to zero, we shift the log losses up to 0 before
    # scaling. This introduces the same multiplicative factor into every
    # probability, which will be fixed by the normalization.
    log_loss_zero_point = max(log_losses.values())
    denormalized_result = {
        hypothesis: prob * 2**(log_losses[hypothesis] - log_loss_zero_point)
        for hypothesis, prob in prior.items()
    }

    total = sum(denormalized_result.values())
    return {h: p / total for h, p in denormalized_result.items()}
