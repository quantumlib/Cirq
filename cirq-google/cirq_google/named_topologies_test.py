from cirq_google.named_topologies import NamedTopology, draw_gridlike, LineTopology, \
    DiagonalRectangleTopology, get_placements, plot_placements


def test_diagonal_rectangle_topology():
    width = 2
    height = 3
    topo = DiagonalRectangleTopology(width, height)
    assert all(1 <= topo.graph.degree[node] <= 4 for node in topo.graph.nodes)
    assert topo.name == '2-3-diagonal-rectangle'
    assert topo.n_nodes == topo.graph.number_of_nodes()


def test_line_topology():
    n = 10
    topo = LineTopology(n)
    assert topo.n_nodes == n
    assert topo.n_nodes == topo.graph.number_of_nodes()
    assert all(1 <= topo.graph.degree[node] <= 2 for node in topo.graph.nodes)
    assert topo.name == '10-line'
