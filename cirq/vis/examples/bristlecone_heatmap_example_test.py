"""Tests for bristlecone_heatmap_example."""

import matplotlib.pyplot as plt

from cirq.vis.examples import bristlecone_heatmap_example


class TestBristleconeHeatmapExample:

    def test_that_example_runs(self):
        plt.switch_backend('agg')
        bristlecone_heatmap_example.main()
