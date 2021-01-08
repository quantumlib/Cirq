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

import cirq


def test_group_settings_greedy_one_group():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    terms = [
        cirq.X(q0),
        cirq.Y(q1),
    ]
    settings = list(cirq.work.observables_to_settings(terms, qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 1

    group_max_obs_should_be = [
        cirq.X(q0) * cirq.Y(q1),
    ]
    group_max_settings_should_be = list(
        cirq.work.observables_to_settings(group_max_obs_should_be, qubits))
    assert set(grouped_settings.keys()) == set(group_max_settings_should_be)

    the_group = grouped_settings[group_max_settings_should_be[0]]
    assert set(the_group) == set(settings)


def test_group_settings_greedy_two_groups():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    terms = [
        cirq.X(q0) * cirq.X(q1),
        cirq.Y(q0) * cirq.Y(q1),
    ]
    settings = list(cirq.work.observables_to_settings(terms, qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 2

    group_max_obs_should_be = terms.copy()
    group_max_settings_should_be = list(
        cirq.work.observables_to_settings(group_max_obs_should_be, qubits))
    assert set(grouped_settings.keys()) == set(group_max_settings_should_be)


def test_group_settings_greedy_single_item():
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits
    term = cirq.X(q0) * cirq.X(q1)

    settings = list(cirq.work.observables_to_settings([term], qubits))
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 1
    assert list(grouped_settings.keys())[0] == settings[0]
    assert list(grouped_settings.values())[0][0] == settings[0]


def test_group_settings_greedy_empty():
    assert cirq.work.group_settings_greedy([]) == dict()


def test_group_settings_greedy_init_state_compat():
    q0, q1 = cirq.LineQubit.range(2)
    settings = [
        cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0) *
                                 cirq.KET_ZERO(q1),
                                 observable=cirq.X(q0)),
        cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0) *
                                 cirq.KET_ZERO(q1),
                                 observable=cirq.Z(q1)),
    ]
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 1


def test_group_settings_greedy_init_state_compat_sparse():
    q0, q1 = cirq.LineQubit.range(2)
    settings = [
        cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0),
                                 observable=cirq.X(q0)),
        cirq.work.InitObsSetting(init_state=cirq.KET_ZERO(q1),
                                 observable=cirq.Z(q1)),
    ]
    grouped_settings = cirq.work.group_settings_greedy(settings)
    # pylint: disable=line-too-long
    grouped_settings_should_be = {
        cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0) * cirq.KET_ZERO(q1),
                                 observable=cirq.X(q0) * cirq.Z(q1)):
        settings
    }
    assert grouped_settings == grouped_settings_should_be


def test_group_settings_greedy_init_state_incompat():
    q0, q1 = cirq.LineQubit.range(2)
    settings = [
        cirq.work.InitObsSetting(init_state=cirq.KET_PLUS(q0) *
                                 cirq.KET_PLUS(q1),
                                 observable=cirq.X(q0)),
        cirq.work.InitObsSetting(init_state=cirq.KET_ZERO(q1),
                                 observable=cirq.Z(q1)),
    ]
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 2


def test_group_settings_greedy_hydrogen():
    qubits = cirq.LineQubit.range(4)
    q0, q1, q2, q3 = qubits
    terms = [
        0.1711977489805745 * cirq.Z(q0), 0.17119774898057447 * cirq.Z(q1),
        -0.2227859302428765 * cirq.Z(q2), -0.22278593024287646 * cirq.Z(q3),
        0.16862219157249939 * cirq.Z(q0) * cirq.Z(q1),
        0.04532220205777764 * cirq.Y(q0) * cirq.X(q1) * cirq.X(q2) * cirq.Y(q3),
        -0.0453222020577776 * cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2) * cirq.X(q3),
        -0.0453222020577776 * cirq.X(q0) * cirq.X(q1) * cirq.Y(q2) * cirq.Y(q3),
        0.04532220205777764 * cirq.X(q0) * cirq.Y(q1) * cirq.Y(q2) * cirq.X(q3),
        0.12054482203290037 * cirq.Z(q0) * cirq.Z(q2), 0.16586702409067802 *
        cirq.Z(q0) * cirq.Z(q3), 0.16586702409067802 * cirq.Z(q1) * cirq.Z(q2),
        0.12054482203290037 * cirq.Z(q1) * cirq.Z(q3),
        0.1743484418396392 * cirq.Z(q2) * cirq.Z(q3)
    ]
    settings = cirq.work.observables_to_settings(terms, qubits)
    grouped_settings = cirq.work.group_settings_greedy(settings)
    assert len(grouped_settings) == 5

    group_max_obs_should_be = [
        cirq.Y(q0) * cirq.X(q1) * cirq.X(q2) * cirq.Y(q3),
        cirq.Y(q0) * cirq.Y(q1) * cirq.X(q2) * cirq.X(q3),
        cirq.X(q0) * cirq.X(q1) * cirq.Y(q2) * cirq.Y(q3),
        cirq.X(q0) * cirq.Y(q1) * cirq.Y(q2) * cirq.X(q3),
        cirq.Z(q0) * cirq.Z(q1) * cirq.Z(q2) * cirq.Z(q3)
    ]
    group_max_settings_should_be = cirq.work.observables_to_settings(
        group_max_obs_should_be, qubits)

    assert set(grouped_settings.keys()) == set(group_max_settings_should_be)
    groups = list(grouped_settings.values())
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 1
    assert len(groups[3]) == 1
    assert len(groups[4]) == len(terms) - 4
