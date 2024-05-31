import numpy as np
import matplotlib.pyplot as plt
import cirq

class HighlightedHeatmap(cirq.Heatmap):
    def __init__(self, error_rates, selected_qubits=None):
        super().__init__(error_rates)
        self.selected_qubits = selected_qubits

    def plot(self, ax=None, collection_options=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        if collection_options is None:
            collection_options = {}
        
        if self.selected_qubits:
            edge_colors = ['red' if q in self.selected_qubits else 'grey' for q in self.qubits]
            linewidths = [4 if q in self.selected_qubits else 2 for q in self.qubits]
            collection_options['linewidths'] = linewidths
            collection_options['edgecolors'] = edge_colors
        
        super().plot(ax=ax, collection_options=collection_options, **kwargs)


class HighlightedTwoQubitInteractionHeatmap(cirq.TwoQubitInteractionHeatmap):
    def __init__(self, interaction_graph, error_rates, selected_qubits=None):
        super().__init__(interaction_graph, error_rates)
        self.selected_qubits = selected_qubits

    def plot(self, ax=None, collection_options=None, **kwargs):
        if ax is None:
            ax = plt.gca()
        
        if collection_options is None:
            collection_options = {}
        
        if self.selected_qubits:
            edge_colors = ['red' if q in self.selected_qubits else 'grey' for q in self.qubits]
            linewidths = [4 if q in self.selected_qubits else 2 for q in self.qubits]
            collection_options['linewidths'] = linewidths
            collection_options['edgecolors'] = edge_colors
        
        super().plot(ax=ax, collection_options=collection_options, **kwargs)


# Example usage
device = cirq.google.Foxtail

# Define selected qubits
selected_qubits = [device.qubits[0], device.qubits[1], device.qubits[2]]

# Create heatmap with selected qubits highlighted
fig, ax = plt.subplots(figsize=(12, 10))
heatmap = HighlightedHeatmap({q: 0.0 for q in device.qubits}, selected_qubits=selected_qubits)
heatmap.plot(ax=ax, collection_options={'cmap': 'binary'}, plot_colorbar=False, annotation_format=None)
heatmap._plot_on_axis(ax)
fig.show()
