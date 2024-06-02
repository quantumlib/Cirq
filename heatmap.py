import matplotlib.pyplot as plt
from cirq.devices import GridQubit
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import PathPatch

# Define the Heatmap class
class Heatmap:
    def __init__(self, value_map, **kwargs):
        self.value_map = value_map
        self.annot_map = {
            (qubit.row, qubit.col): format(value, '.2g')
            for qubit, value in value_map.items()
        }
        self._config = {
            "plot_colorbar": True,
            "colorbar_position": "right",
            "colorbar_size": "5%",
            "colorbar_pad": "2%",
            "colormap": "viridis",
            "annotation_format": ".2g",
        }
        self._config.update(kwargs)
        self.annot_kwargs = {}

    def set_annotation_map(self, annot_map, **text_options):
        self.annot_map = annot_map
        self.annot_kwargs = text_options
        return self

    def set_annotation_format(self, annot_format, **text_options):
        self.annot_map = {
            (qubit.row, qubit.col): format(value, annot_format)
            for qubit, value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def set_colorbar(self, position='right', size='5%', pad='2%', **colorbar_options):
        self._config['plot_colorbar'] = True
        self._config['colorbar_position'] = position
        self._config['colorbar_size'] = size
        self._config['colorbar_pad'] = pad
        self._config['colorbar_options'] = colorbar_options
        return self

    def unset_colorbar(self):
        self._config['plot_colorbar'] = False
        return self

    def plot(self, ax=None, selected_qubits=[], **collection_options):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))

        qubit_coords = [(qubit.row, qubit.col) for qubit in self.value_map.keys()]
        min_row, max_row = min(row for row, _ in qubit_coords), max(row for row, _ in qubit_coords)
        min_col, max_col = min(col for _, col in qubit_coords), max(col for _, col in qubit_coords)

        value_table = pd.DataFrame(
            np.nan, index=range(min_row, max_row + 1), columns=range(min_col, max_col + 1)
        )
        for qubit, value in self.value_map.items():
            value_table.loc[qubit.row, qubit.col] = value

        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)] * (max_row - min_row + 2))
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)] * (max_col - min_col + 2)).transpose()

        collection_options_filtered = {k: v for k, v in collection_options.items() if k not in self._config}

        mesh = ax.pcolor(
            x_table,
            y_table,
            value_table,
            vmin=self._config.get('vmin', None),
            vmax=self._config.get('vmax', None),
            cmap=self._config['colormap'],
            **collection_options_filtered
        )

        edge_colors = ['red' if GridQubit(row, col) in selected_qubits else 'grey' for row, col in qubit_coords]
        linewidths = [4 if GridQubit(row, col) in selected_qubits else 2 for row, col in qubit_coords]
        linestyles = ['solid' if GridQubit(row, col) in selected_qubits else 'dashed' for row, col in qubit_coords]

        for path, edgecolor, linewidth, linestyle in zip(mesh.get_paths(), edge_colors, linewidths, linestyles):
            pathpatch = PathPatch(path, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle)
            ax.add_patch(pathpatch)

        if self._config['plot_colorbar']:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(mesh, ax)

        if show_plot:
            plt.show()

        return ax, mesh, value_table

    def _plot_colorbar(self, mappable, ax):
        position = self._config['colorbar_position']
        size = self._config['colorbar_size']
        pad = self._config['colorbar_pad']
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes(position, size=size, pad=pad)
        orientation = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = ax.figure.colorbar(mappable, cax=colorbar_ax, orientation=orientation)
        colorbar_ax.tick_params(axis='y', direction='out')

    def _write_annotations(self, mesh, ax):
        for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
            vertices = path.vertices[:4]
            row = int(np.mean([v[1] for v in vertices]))
            col = int(np.mean([v[0] for v in vertices]))
            annotation = self.annot_map.get((row, col), '')
            if annotation:
                luminance = 0.2126 * facecolor[0] + 0.7152 * facecolor[1] + 0.0722 * facecolor[2]
                text_color = 'black' if luminance > 0.4 else 'white'
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kwargs)
                ax.text(col, row, annotation, **text_kwargs)

# Create a list of GridQubit instances for demonstration
qubits = [GridQubit(row, col) for row in range(3) for col in range(3)]

# Example usage of the modified Heatmap class
fig, ax = plt.subplots(figsize=(12, 10))
selected_qubits = [GridQubit(1, 1), GridQubit(2, 2)]
heatmap = Heatmap({q: 0.0 for q in qubits})
heatmap.plot(ax=ax, selected_qubits=selected_qubits, plot_colorbar=False, annotation_format=None)
plt.show()



import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cirq.devices import GridQubit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import PathPatch

class TwoQubitInteractionHeatmap:
    def __init__(self, value_map, **kwargs):
        self.value_map = value_map
        self.annot_map = {
            (q1.row, q1.col, q2.row, q2.col): format(value, '.2g')
            for (q1, q2), value in value_map.items()
        }
        self._config = {
            "plot_colorbar": True,
            "colorbar_position": "right",
            "colorbar_size": "5%",
            "colorbar_pad": "2%",
            "colormap": "viridis",
            "annotation_format": ".2g",
        }
        self._config.update(kwargs)
        self.annot_kwargs = {}

    def set_annotation_map(self, annot_map, **text_options):
        self.annot_map = annot_map
        self.annot_kwargs = text_options
        return self

    def set_annotation_format(self, annot_format, **text_options):
        self.annot_map = {
            (q1.row, q1.col, q2.row, q2.col): format(value, annot_format)
            for (q1, q2), value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def set_colorbar(self, position='right', size='5%', pad='2%', **colorbar_options):
        self._config['plot_colorbar'] = True
        self._config['colorbar_position'] = position
        self._config['colorbar_size'] = size
        self._config['colorbar_pad'] = pad
        self._config['colorbar_options'] = colorbar_options
        return self

    def unset_colorbar(self):
        self._config['plot_colorbar'] = False
        return self

    def plot(self, ax=None, selected_qubits=[], **collection_options):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))

        qubit_coords = [(q1.row, q1.col, q2.row, q2.col) for (q1, q2) in self.value_map.keys()]
        min_row, max_row = min(row for row, _, _, _ in qubit_coords), max(row for row, _, _, _ in qubit_coords)
        min_col, max_col = min(col for _, col, _, _ in qubit_coords), max(col for _, col, _, _ in qubit_coords)

        value_table = pd.DataFrame(
            np.nan, index=range(min_row, max_row + 1), columns=range(min_col, max_col + 1)
        )
        for (q1, q2), value in self.value_map.items():
            value_table.loc[q1.row, q1.col] = value
            value_table.loc[q2.row, q2.col] = value

        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)] * (max_row - min_row + 2))
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)] * (max_col - min_col + 2)).transpose()

        collection_options_filtered = {k: v for k, v in collection_options.items() if k not in self._config}

        mesh = ax.pcolor(
            x_table,
            y_table,
            value_table,
            vmin=self._config.get('vmin', None),
            vmax=self._config.get('vmax', None),
            cmap=self._config['colormap'],
            **collection_options_filtered
        )

        edge_colors = ['red' if q1 in selected_qubits or q2 in selected_qubits else 'grey' for q1, _, q2, _ in qubit_coords]
        linewidths = [4 if q1 in selected_qubits or q2 in selected_qubits else 2 for q1, _, q2, _ in qubit_coords]
        linestyles = ['solid' if q1 in selected_qubits or q2 in selected_qubits else 'dashed' for q1, _, q2, _ in qubit_coords]

        for path, edgecolor, linewidth, linestyle in zip(mesh.get_paths(), edge_colors, linewidths, linestyles):
            pathpatch = PathPatch(path, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle)
            ax.add_patch(pathpatch)

        if self._config['plot_colorbar']:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(mesh, ax)

        if show_plot:
            plt.show()

        return ax, mesh, value_table

    def _plot_colorbar(self, mappable, ax):
        position = self._config['colorbar_position']
        size = self._config['colorbar_size']
        pad = self._config['colorbar_pad']
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes(position, size=size, pad=pad)
        orientation = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = ax.figure.colorbar(mappable, cax=colorbar_ax, orientation=orientation)
        colorbar_ax.tick_params(axis='y', direction='out')

    def _write_annotations(self, mesh, ax):
        for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
            vertices = path.vertices[:4]
            row1 = int(np.mean([v[1] for v in vertices[:2]]))
            col1 = int(np.mean([v[0] for v in vertices[:2]]))
            row2 = int(np.mean([v[1] for v in vertices[2:]]))
            col2 = int(np.mean([v[0] for v in vertices[2:]]))
            annotation = self.annot_map.get((row1, col1, row2, col2), '')
            if annotation:
                luminance = 0.2126 * facecolor[0] + 0.7152 * facecolor[1] + 0.0722 * facecolor[2]
                text_color = 'black' if luminance > 0.4 else 'white'
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kwargs)
                ax.text((col1 + col2) / 2, (row1 + row2) / 2, annotation, **text_kwargs)



# Define  value_map dictionary with qubit pairs as keys
value_map = {
    (GridQubit(1, 1), GridQubit(2, 2)): 0.0,
    (GridQubit(3, 3), GridQubit(4, 4)): 0.0,
    #  more qubit pairs as needed
}

# Example usage of the modified TwoQubitInteractionHeatmap class
fig, ax = plt.subplots(figsize=(12, 10))
selected_qubits = [GridQubit(1, 1), GridQubit(2, 2)]  # Adjust this as needed
heatmap = TwoQubitInteractionHeatmap(value_map)
heatmap.plot(ax=ax, selected_qubits=selected_qubits, plot_colorbar=False, annotation_format=None)
fig.show()



from matplotlib.patches import PathPatch
from cirq.devices import GridQubit
class TwoQubitInteractionHeatmap:
    def __init__(self, value_map, **kwargs):
        self.value_map = value_map
        self.annot_map = {
            (q1.row, q1.col, q2.row, q2.col): format(value, '.2g')
            for (q1, q2), value in value_map.items()
        }
        self._config = {
            "plot_colorbar": True,
            "colorbar_position": "right",
            "colorbar_size": "5%",
            "colorbar_pad": "2%",
            "colormap": "viridis",
            "annotation_format": ".2g",
        }
        self._config.update(kwargs)
        self.annot_kwargs = {}

    def set_annotation_map(self, annot_map, **text_options):
        self.annot_map = annot_map
        self.annot_kwargs = text_options
        return self

    def set_annotation_format(self, annot_format, **text_options):
        self.annot_map = {
            (q1.row, q1.col, q2.row, q2.col): format(value, annot_format)
            for (q1, q2), value in self.value_map.items()
        }
        self.annot_kwargs = text_options
        return self

    def set_colorbar(self, position='right', size='5%', pad='2%', **colorbar_options):
        self._config['plot_colorbar'] = True
        self._config['colorbar_position'] = position
        self._config['colorbar_size'] = size
        self._config['colorbar_pad'] = pad
        self._config['colorbar_options'] = colorbar_options
        return self

    def unset_colorbar(self):
        self._config['plot_colorbar'] = False
        return self

    def plot(self, ax=None, selected_qubits=[], highlight_qubits=[], **collection_options):
        show_plot = not ax
        if not ax:
            fig, ax = plt.subplots(figsize=(8, 8))

        qubit_coords = [(q1.row, q1.col, q2.row, q2.col) for (q1, q2) in self.value_map.keys()]
        min_row, max_row = min(row for row, _, _, _ in qubit_coords), max(row for row, _, _, _ in qubit_coords)
        min_col, max_col = min(col for _, col, _, _ in qubit_coords), max(col for _, col, _, _ in qubit_coords)

        value_table = pd.DataFrame(
            np.nan, index=range(min_row, max_row + 1), columns=range(min_col, max_col + 1)
        )
        for (q1, q2), value in self.value_map.items():
            value_table.loc[q1.row, q1.col] = value
            value_table.loc[q2.row, q2.col] = value

        x_table = np.array([np.arange(min_col - 0.5, max_col + 1.5)] * (max_row - min_row + 2))
        y_table = np.array([np.arange(min_row - 0.5, max_row + 1.5)] * (max_col - min_col + 2)).transpose()

        collection_options_filtered = {k: v for k, v in collection_options.items() if k not in self._config}

        mesh = ax.pcolor(
            x_table,
            y_table,
            value_table,
            vmin=self._config.get('vmin', None),
            vmax=self._config.get('vmax', None),
            cmap=self._config['colormap'],
            **collection_options_filtered
        )

        edge_colors = ['red' if q1 in selected_qubits or q2 in selected_qubits else 'grey' for q1, _, q2, _ in qubit_coords]
        linewidths = [4 if q1 in selected_qubits or q2 in selected_qubits else 2 for q1, _, q2, _ in qubit_coords]
        linestyles = ['solid' if q1 in selected_qubits or q2 in selected_qubits else 'dashed' for q1, _, q2, _ in qubit_coords]

        for path, edgecolor, linewidth, linestyle in zip(mesh.get_paths(), edge_colors, linewidths, linestyles):
            pathpatch = PathPatch(path, edgecolor=edgecolor, linewidth=linewidth, linestyle=linestyle)
            ax.add_patch(pathpatch)

        # Highlight qubits
        for q in highlight_qubits:
            ax.add_patch(plt.Rectangle((q.col - 0.5, q.row - 0.5), 1, 1, fill=False, edgecolor='blue', linewidth=2))

        if self._config['plot_colorbar']:
            self._plot_colorbar(mesh, ax)

        if self.annot_map:
            self._write_annotations(mesh, ax)

        if show_plot:
            plt.show()

        return ax, mesh, value_table

    def _plot_colorbar(self, mappable, ax):
        position = self._config['colorbar_position']
        size = self._config['colorbar_size']
        pad = self._config['colorbar_pad']
        divider = make_axes_locatable(ax)
        colorbar_ax = divider.append_axes(position, size=size, pad=pad)
        orientation = 'vertical' if position in ('left', 'right') else 'horizontal'
        colorbar = ax.figure.colorbar(mappable, cax=colorbar_ax, orientation=orientation)
        colorbar_ax.tick_params(axis='y', direction='out')

    def _write_annotations(self, mesh, ax):
        for path, facecolor in zip(mesh.get_paths(), mesh.get_facecolors()):
            vertices = path.vertices[:4]
            row1 = int(np.mean([v[1] for v in vertices[:2]]))
            col1 = int(np.mean([v[0] for v in vertices[:2]]))
            row2 = int(np.mean([v[1] for v in vertices[2:]]))
            col2 = int(np.mean([v[0] for v in vertices[2:]]))
            annotation = self.annot_map.get((row1, col1, row2, col2), '')
            if annotation:
                luminance = 0.2126 * facecolor[0] + 0.7152 * facecolor[1] + 0.0722 * facecolor[2]
                text_color = 'black' if luminance > 0.4 else 'white'
                text_kwargs = dict(color=text_color, ha="center", va="center")
                text_kwargs.update(self.annot_kwargs)
                ax.text((col1 + col2) / 2, (row1 + row2) / 2, annotation, **text_kwargs)


# Define  value_map dictionary with qubit pairs as keys
value_map = {
    (GridQubit(1, 1), GridQubit(2, 2)): 0.0,
    (GridQubit(3, 3), GridQubit(4, 4)): 0.0,
    # more qubit pairs as needed
}

# Example usage of the modified TwoQubitInteractionHeatmap class
fig, ax = plt.subplots(figsize=(12, 10))
selected_qubits = [GridQubit(1, 1), GridQubit(2, 2)]  # Adjust this as needed
highlight_qubits = [GridQubit(3, 3)]  # Adjust this as needed
heatmap = TwoQubitInteractionHeatmap(value_map)
heatmap.plot(ax=ax, selected_qubits=selected_qubits, highlight_qubits=highlight_qubits, plot_colorbar=False, annotation_format=None)
fig.show()
