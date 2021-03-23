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
"""Tests for Histogram."""

import numpy as np
import pytest

import matplotlib.pyplot as plt

from cirq.vis import integrated_histogram


@pytest.mark.parametrize('data', [range(10), {f'key_{i}': i for i in range(10)}])
def test_integrated_histogram(data):
    ax = integrated_histogram(
        data,
        title='Test Plot',
        axis_label='Y Axis Label',
        color='r',
        label='line label',
        cdf_on_x=True,
        show_zero=True,
    )
    assert ax.get_title() == 'Test Plot'
    assert ax.get_ylabel() == 'Y Axis Label'
    assert len(ax.get_lines()) == 2
    for line in ax.get_lines():
        assert line.get_color() == 'r'


def test_multiple_plots():
    _, ax = plt.subplots(1, 1)
    n = 53
    data = np.random.random_sample((2, n))
    integrated_histogram(
        data[0],
        ax,
        color='r',
        label='data_1',
        median_line=False,
        mean_line=True,
        mean_label='mean_1',
    )
    integrated_histogram(data[1], ax, color='k', label='data_2', median_label='median_2')
    assert ax.get_title() == 'N=53'
    for line in ax.get_lines():
        assert line.get_color() in ['r', 'k']
        assert line.get_label() in ['data_1', 'data_2', 'mean_1', 'median_2']
