# Copyright 2019 The Cirq Developers
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
"""Tests for Heatmap."""

import pathlib
import string
from tempfile import mkdtemp

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt

from cirq.devices import grid_qubit
from cirq.vis import heatmap


@pytest.fixture
def ax():
    figure = mpl.figure.Figure()
    return figure.add_subplot(111)


@pytest.mark.parametrize('test_GridQubit', [True, False])
def test_cells_positions(ax, test_GridQubit):
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    if test_GridQubit:
        qubits = [grid_qubit.GridQubit(row, col) for (row, col) in row_col_list]
    else:
        qubits = row_col_list
    values = np.random.random(len(qubits))
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    _, mesh, _ = heatmap.Heatmap(test_value_map).plot(ax)

    found_qubits = set()
    for path in mesh.get_paths():
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        found_qubits.add((row, col))
    assert found_qubits == set(row_col_list)


# Test colormaps are the first one in each category in
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html.
@pytest.mark.parametrize(
    'colormap_name', ['viridis', 'Greys', 'binary', 'PiYG', 'twilight', 'Pastel1', 'flag']
)
def test_cell_colors(ax, colormap_name):
    qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    values = 1.0 + 2.0 * np.random.random(len(qubits))  # [1, 3)
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    vmin, vmax = 1.5, 2.5
    random_heatmap = heatmap.Heatmap(test_value_map).set_colormap(
        colormap_name, vmin=vmin, vmax=vmax
    )
    _, mesh, _ = random_heatmap.plot(ax)

    colormap = mpl.cm.get_cmap(colormap_name)
    for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        value = test_value_map[(row, col)]
        color_scale = (value - vmin) / (vmax - vmin)
        if color_scale < 0.0:
            color_scale = 0.0
        if color_scale > 1.0:
            color_scale = 1.0
        expected_color = np.array(colormap(color_scale))
        assert np.all(np.isclose(facecolor, expected_color))


def test_default_annotation(ax):
    """Tests that the default annotation is '.2g' format on float(value)."""
    qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    values = ['3.752', '42', '-5.27e8', '-7.34e-9', 732, 0.432, 3.9753e28]
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    random_heatmap = heatmap.Heatmap(test_value_map)
    random_heatmap.plot(ax)
    actual_texts = set()
    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            text = artist.get_text()
            actual_texts.add(((row, col), text))
    expected_texts = set(
        (qubit, format(float(value), '.2g')) for qubit, value in test_value_map.items()
    )
    assert expected_texts.issubset(actual_texts)


@pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g'])
def test_annotation_position_and_content(ax, format_string):
    qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    values = np.random.random(len(qubits))
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    random_heatmap = heatmap.Heatmap(test_value_map).set_annotation_format(format_string)
    random_heatmap.plot(ax)
    actual_texts = set()
    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            text = artist.get_text()
            actual_texts.add(((row, col), text))
    expected_texts = set(
        (qubit, format(value, format_string)) for qubit, value in test_value_map.items()
    )
    assert expected_texts.issubset(actual_texts)


@pytest.mark.parametrize('test_GridQubit', [True, False])
def test_annotation_map(ax, test_GridQubit):
    row_col_list = [(0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8)]
    if test_GridQubit:
        qubits = [grid_qubit.GridQubit(*row_col) for row_col in row_col_list]
    else:
        qubits = row_col_list
    values = np.random.random(len(qubits))
    annos = np.random.choice([c for c in string.ascii_letters], len(qubits))
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    test_anno_map = {
        qubit: anno
        for qubit, row_col, anno in zip(qubits, row_col_list, annos)
        if row_col != (1, 6)
    }
    random_heatmap = heatmap.Heatmap(test_value_map).set_annotation_map(test_anno_map)
    random_heatmap.plot(ax)
    actual_texts = set()
    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            assert (row, col) != (1, 6)
            actual_texts.add(((row, col), artist.get_text()))
    expected_texts = set(
        (row_col, anno) for row_col, anno in zip(row_col_list, annos) if row_col != (1, 6)
    )
    assert expected_texts.issubset(actual_texts)


@pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g', 's'])
def test_non_float_values(ax, format_string):
    class Foo:
        def __init__(self, value: float, unit: str):
            self.value = value
            self.unit = unit

        def __float__(self):
            return self.value

        def __format__(self, format_string):
            if format_string == 's':
                return f'{self.value}{self.unit}'
            else:
                return format(self.value, format_string)

    qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    values = np.random.random(len(qubits))
    units = np.random.choice([c for c in string.ascii_letters], len(qubits))
    test_value_map = {
        qubit: Foo(float(value), unit) for qubit, value, unit in zip(qubits, values, units)
    }
    colormap_name = 'viridis'
    vmin, vmax = 0.0, 1.0
    random_heatmap = (
        heatmap.Heatmap(test_value_map)
        .set_colormap(colormap_name, vmin=vmin, vmax=vmax)
        .set_annotation_format(format_string)
    )
    _, mesh, _ = random_heatmap.plot(ax)

    colormap = mpl.cm.get_cmap(colormap_name)
    for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
        vertices = path.vertices[0:4]
        row = int(round(np.mean([v[1] for v in vertices])))
        col = int(round(np.mean([v[0] for v in vertices])))
        foo = test_value_map[(row, col)]
        color_scale = (foo.value - vmin) / (vmax - vmin)
        expected_color = np.array(colormap(color_scale))
        assert np.all(np.isclose(facecolor, expected_color))

    for artist in ax.get_children():
        if isinstance(artist, mpl.text.Text):
            col, row = artist.get_position()
            if (row, col) in test_value_map:
                foo = test_value_map[(row, col)]
                actual_text = artist.get_text()
                expected_text = format(foo, format_string)
                assert actual_text == expected_text


@pytest.mark.parametrize('test_GridQubit', [True, False])
def test_urls(ax, test_GridQubit):
    row_col_list = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    if test_GridQubit:
        qubits = [grid_qubit.GridQubit(*row_col) for row_col in row_col_list]
    else:
        qubits = row_col_list
    values = np.random.random(len(qubits))
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    test_url_map = {
        qubit: 'http://google.com/{}+{}'.format(*row_col)
        for qubit, row_col in zip(qubits, row_col_list)
        if row_col != (1, 6)
    }
    # Add an extra entry that should not show up in results because the
    # qubit is not in the value map.
    extra_qubit = grid_qubit.GridQubit(10, 7) if test_GridQubit else (10, 7)
    test_url_map[extra_qubit] = 'http://google.com/10+7'

    my_heatmap = heatmap.Heatmap(test_value_map).set_url_map(test_url_map)
    _, mesh, _ = my_heatmap.plot(ax)
    expected_urls = [
        test_url_map.get(qubit, '') for row_col, qubit in sorted(zip(row_col_list, qubits))
    ]
    assert mesh.get_urls() == expected_urls


@pytest.mark.parametrize(
    'position,size,pad',
    [
        ('right', "5%", "2%"),
        ('right', "5%", "10%"),
        ('right', "20%", "2%"),
        ('right', "20%", "10%"),
        ('left', "5%", "2%"),
        ('left', "5%", "10%"),
        ('left', "20%", "2%"),
        ('left', "20%", "10%"),
        ('top', "5%", "2%"),
        ('top', "5%", "10%"),
        ('top', "20%", "2%"),
        ('top', "20%", "10%"),
        ('bottom', "5%", "2%"),
        ('bottom', "5%", "10%"),
        ('bottom', "20%", "2%"),
        ('bottom', "20%", "10%"),
    ],
)
def test_colorbar(ax, position, size, pad):
    qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
    values = np.random.random(len(qubits))
    test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
    random_heatmap = heatmap.Heatmap(test_value_map).unset_colorbar()
    fig1, ax1 = plt.subplots()
    random_heatmap.plot(ax1)
    random_heatmap.set_colorbar(position=position, size=size, pad=pad)
    fig2, ax2 = plt.subplots()
    random_heatmap.plot(ax2)

    # We need to call savefig() explicitly for updating axes position since the figure
    # object has been altered in the HeatMap._plot_colorbar function.
    tmp_dir = mkdtemp()
    fig2.savefig(pathlib.Path(tmp_dir) / 'tmp.png')

    # Check that the figure has one more object in it when colorbar is on.
    assert len(fig2.get_children()) == len(fig1.get_children()) + 1

    fig_pos = fig2.get_axes()[0].get_position()
    colorbar_pos = fig2.get_axes()[1].get_position()

    origin_axes_size = (
        fig_pos.xmax - fig_pos.xmin
        if position in ["left", "right"]
        else fig_pos.ymax - fig_pos.ymin
    )
    expected_pad = int(pad.replace("%", "")) / 100 * origin_axes_size
    expected_size = int(size.replace("%", "")) / 100 * origin_axes_size

    if position == "right":
        pad_distance = colorbar_pos.xmin - fig_pos.xmax
        colorbar_size = colorbar_pos.xmax - colorbar_pos.xmin
    elif position == "left":
        pad_distance = fig_pos.xmin - colorbar_pos.xmax
        colorbar_size = colorbar_pos.xmax - colorbar_pos.xmin
    elif position == "top":
        pad_distance = colorbar_pos.ymin - fig_pos.ymax
        colorbar_size = colorbar_pos.ymax - colorbar_pos.ymin
    elif position == "bottom":
        pad_distance = fig_pos.ymin - colorbar_pos.ymax
        colorbar_size = colorbar_pos.ymax - colorbar_pos.ymin

    assert np.isclose(colorbar_size, expected_size)
    assert np.isclose(pad_distance, expected_pad)

    plt.close(fig1)
    plt.close(fig2)
