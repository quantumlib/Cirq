"""Tests for heatmap."""

import string

import numpy as np
import pytest

import matplotlib as mpl
import matplotlib.pyplot as plt

from cirq.google.vis import heatmap


class TestHeatmap:

    def test_cells_positions(self):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = np.random.random(len(qubits))
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        _, ax = plt.subplots()
        mesh, _ = heatmap.Heatmap(test_value_map).plot(ax)

        found_qubits = set()
        for path in mesh.get_paths():
            vertices = path.vertices[0:4]
            row = int(round(np.mean([v[1] for v in vertices])))
            col = int(round(np.mean([v[0] for v in vertices])))
            found_qubits.add((row, col))
        assert found_qubits == set(qubits)

    # Test colormaps are the first one in each category in
    # https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html.
    @pytest.mark.parametrize(
        'colormap_name',
        ['viridis', 'Greys', 'binary', 'PiYG', 'twilight', 'Pastel1', 'flag'])
    def test_cell_colors(self, colormap_name):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = 1.0 + 2.0 * np.random.random(len(qubits))  # [1, 3)
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        vmin, vmax = 1.5, 2.5
        random_heatmap = (heatmap.Heatmap(test_value_map).set_colormap(
            colormap_name, vmin=vmin, vmax=vmax))
        _, ax = plt.subplots()
        mesh, _ = random_heatmap.plot(ax)

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

    @pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g'])
    def test_annotation_position_and_content(self, format_string):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = np.random.random(len(qubits))
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        random_heatmap = (heatmap.Heatmap(test_value_map).set_annotation_format(
            format_string))
        _, ax = plt.subplots()
        random_heatmap.plot(ax)
        actual_texts = set()
        for artist in ax.get_children():
            if isinstance(artist, mpl.text.Text):
                col, row = artist.get_position()
                text = artist.get_text()
                actual_texts.add(((row, col), text))
        expected_texts = set((qubit, format(value, format_string))
                             for qubit, value in test_value_map.items())
        assert expected_texts.issubset(actual_texts)

    def test_annotation_map(self):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = np.random.random(len(qubits))
        annos = np.random.choice([c for c in string.ascii_letters], len(qubits))
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        test_anno_map = {
            qubit: anno for qubit, anno in zip(qubits, annos) if qubit != (1, 6)
        }
        random_heatmap = (
            heatmap.Heatmap(test_value_map).set_annotation_map(test_anno_map))
        _, ax = plt.subplots()
        random_heatmap.plot(ax)
        actual_texts = set()
        for artist in ax.get_children():
            if isinstance(artist, mpl.text.Text):
                col, row = artist.get_position()
                assert (row, col) != (1, 6)
                actual_texts.add(((row, col), artist.get_text()))
        expected_texts = set(test_anno_map.items())
        assert expected_texts.issubset(actual_texts)

    @pytest.mark.parametrize('format_string', ['.3e', '.2f', '.4g', 's'])
    def test_non_float_values(self, format_string):

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
            qubit: Foo(float(value), unit)
            for qubit, value, unit in zip(qubits, values, units)
        }
        colormap_name = 'viridis'
        vmin, vmax = 0.0, 1.0
        random_heatmap = (heatmap.Heatmap(test_value_map).set_colormap(
            colormap_name, vmin=vmin,
            vmax=vmax).set_annotation_format(format_string))
        _, ax = plt.subplots()
        mesh, _ = random_heatmap.plot(ax)

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

    def test_urls(self):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = np.random.random(len(qubits))
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        test_url_map = {
            qubit: 'http://google.com/{}+{}'.format(qubit[0], qubit[1])
            for qubit in qubits
            if qubit != (1, 6)
        }
        # Add an extra entry that should not show up in results because the
        # qubit is not in the value map.
        test_url_map[(10, 7)] = 'http://google.com/10+7'

        _, ax = plt.subplots()
        mesh, _ = (
            heatmap.Heatmap(test_value_map).set_url_map(test_url_map)).plot(ax)
        expected_urls = [
            test_url_map.get((row, col), '') for row, col in sorted(qubits)
        ]
        assert mesh.get_urls() == expected_urls

    def test_colorbar(self):
        qubits = ((0, 5), (8, 1), (7, 0), (13, 5), (1, 6), (3, 2), (2, 8))
        values = np.random.random(len(qubits))
        test_value_map = {qubit: value for qubit, value in zip(qubits, values)}
        random_heatmap = heatmap.Heatmap(test_value_map).unset_colorbar()
        fig1, ax1 = plt.subplots()
        random_heatmap.plot(ax1)
        random_heatmap.set_colorbar()
        fig2, ax2 = plt.subplots()
        random_heatmap.plot(ax2)

        # Check that the figure has one more object in it when colorbar is on.
        assert len(fig2.get_children()) == len(fig1.get_children()) + 1

        # TODO: Make this is a more thorough test, e.g., we should test that the
        # position, size and pad arguments are respected.
