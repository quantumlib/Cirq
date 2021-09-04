# Copyright 2020 The Cirq developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Iterable, Dict, List, TYPE_CHECKING, cast, Callable

from cirq import ops, value
from cirq.work.observable_settings import InitObsSetting, _max_weight_state, _max_weight_observable

if TYPE_CHECKING:
    pass

GROUPER_T = Callable[[Iterable[InitObsSetting]], Dict[InitObsSetting, List[InitObsSetting]]]


def group_settings_greedy(
    settings: Iterable[InitObsSetting],
) -> Dict[InitObsSetting, List[InitObsSetting]]:
    """Greedily group settings which can be simultaneously measured.

    We construct a dictionary keyed by `max_setting` (see docstrings
    for `_max_weight_state` and `_max_weight_observable`) where the value
    is a list of settings compatible with `max_setting`. For each new setting,
    we try to find an existing group to add it and update `max_setting` for
    that group if necessary. Otherwise, we make a new group.

    In practice, this greedy algorithm performs comparably to something
    more complicated by solving the clique cover problem on a graph
    of simultaneously-measurable settings.

    Args:
        settings: The settings to group.

    Returns:
        A dictionary keyed by `max_setting` which need not exist in the
        input list of settings. Each dictionary value is a list of
        settings compatible with `max_setting`.
    """
    grouped_settings = {}  # type: Dict[InitObsSetting, List[InitObsSetting]]
    for setting in settings:
        for max_setting, simul_settings in grouped_settings.items():
            trial_grouped_settings = simul_settings + [setting]
            new_max_weight_state = _max_weight_state(
                stg.init_state for stg in trial_grouped_settings
            )
            new_max_weight_obs = _max_weight_observable(
                stg.observable for stg in trial_grouped_settings
            )
            compatible_init_state = new_max_weight_state is not None
            compatible_observable = new_max_weight_obs is not None
            can_be_inserted = compatible_init_state and compatible_observable
            if can_be_inserted:
                new_max_weight_state = cast(value.ProductState, new_max_weight_state)
                new_max_weight_obs = cast(ops.PauliString, new_max_weight_obs)
                del grouped_settings[max_setting]
                new_max_setting = InitObsSetting(new_max_weight_state, new_max_weight_obs)
                grouped_settings[new_max_setting] = trial_grouped_settings
                break

        else:
            # made it through entire dict without finding a compatible group,
            # thus a new group needs to be created
            # Strip coefficients before using as key
            new_max_weight_obs = setting.observable.with_coefficient(1.0)
            new_max_setting = InitObsSetting(setting.init_state, new_max_weight_obs)
            grouped_settings[new_max_setting] = [setting]

    return grouped_settings
