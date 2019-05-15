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
"""Defines sweep trial results."""

from typing import Dict, Any, TYPE_CHECKING, List, Set

from cirq import value
from cirq.study import resolver, trial_result

if TYPE_CHECKING:
    # pylint: disable=unused-import
    import cirq


@value.value_equality(unhashable=True)
class SweepTrialResult:
    """The combination of multiple TrialResults with potentially different
       set of parameters.

    Attributes:
        trial_results: A list of TrialResults.
    """

    def __init__(self,
                 trial_results: List[trial_result.TrialResult] = None) -> None:
        """
        Args:
            params: A ParamResolver of settings used for this result.
            measurements: A dictionary from measurement gate key to measurement
                results. The value for each key is a 2-D array of booleans,
                with the first index running over the repetitions, and the
                second index running over the qubits for the corresponding
                measurements.
            repetitions: The number of times the circuit was sampled.
        """
        self.trial_results = trial_results if trial_results is not None else []
        self._index = {}  # type: Dict[str, Set[int]]
        for i, result in enumerate(self.trial_results):
            for key, value in result.params.param_dict.items():
                self._index.setdefault(key + str(value), set()).add(i)

    def trials_where_params_match(
            self, param_dict: resolver.ParamResolverOrSimilarType
    ) -> List[trial_result.TrialResult]:
        """
        Gets the list of TrialResults that match the input mapping of
        parameters to their values. Takes an "and" of all the specified
        conditions.

        Args:
            param_dict: A dictionary from the ParameterValue key (str) to its
                assigned value.
        Returns:
            A list of TrialResults.
        """
        trial_results_match = set(range(len(self.trial_results)))
        for key, value in resolver.ParamResolver(param_dict).param_dict.items():
            trial_results_match.intersection_update(
                self._index.get(key + str(value), set()))
        return [self.trial_results[i] for i in trial_results_match]

    def slice_where_params_match(self,
                                 param_dict: resolver.ParamResolverOrSimilarType
                                ) -> 'cirq.SweepTrialResult':
        """
        Gets a SweepTrialResult which represents a slice of the current one
        limited to the TrialResults that match the input mapping of parameters
        to their values. Takes an "and" of all the specified conditions.

        Args:
            param_dict: A dictionary from the ParameterValue key (str) to its
                assigned value.
        Returns:
            A SweepTrialResult limited to TrialResults that match the specified
            input.
        """
        return SweepTrialResult(self.trials_where_params_match(param_dict))

    def __repr__(self):
        return ('cirq.SweepTrialResult(trial_results={!r})').format(
            self.trial_results)

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        """Output to show in ipython and Jupyter notebooks."""
        if cycle:
            # There should never be a cycle.  This is just in case.
            p.text('SweepTrialResult(...)')
        else:
            p.text(str(self))

    def __str__(self):
        return '[{' + '}, {'.join(
            [str(trial_result) for trial_result in self.trial_results]) + '}]'

    def _value_equality_values_(self):
        return sorted([repr(result) for result in self.trial_results])
