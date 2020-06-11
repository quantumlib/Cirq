<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.CircuitDag" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="adjlist_inner_dict_factory"/>
<meta itemprop="property" content="adjlist_outer_dict_factory"/>
<meta itemprop="property" content="edge_attr_dict_factory"/>
<meta itemprop="property" content="graph_attr_dict_factory"/>
<meta itemprop="property" content="node_attr_dict_factory"/>
<meta itemprop="property" content="node_dict_factory"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="add_edge"/>
<meta itemprop="property" content="add_edges_from"/>
<meta itemprop="property" content="add_node"/>
<meta itemprop="property" content="add_nodes_from"/>
<meta itemprop="property" content="add_weighted_edges_from"/>
<meta itemprop="property" content="adjacency"/>
<meta itemprop="property" content="all_operations"/>
<meta itemprop="property" content="all_qubits"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="disjoint_qubits"/>
<meta itemprop="property" content="edge_subgraph"/>
<meta itemprop="property" content="findall_nodes_until_blocked"/>
<meta itemprop="property" content="from_circuit"/>
<meta itemprop="property" content="from_ops"/>
<meta itemprop="property" content="get_edge_data"/>
<meta itemprop="property" content="has_edge"/>
<meta itemprop="property" content="has_node"/>
<meta itemprop="property" content="has_predecessor"/>
<meta itemprop="property" content="has_successor"/>
<meta itemprop="property" content="is_directed"/>
<meta itemprop="property" content="is_multigraph"/>
<meta itemprop="property" content="make_node"/>
<meta itemprop="property" content="nbunch_iter"/>
<meta itemprop="property" content="neighbors"/>
<meta itemprop="property" content="number_of_edges"/>
<meta itemprop="property" content="number_of_nodes"/>
<meta itemprop="property" content="order"/>
<meta itemprop="property" content="ordered_nodes"/>
<meta itemprop="property" content="predecessors"/>
<meta itemprop="property" content="remove_edge"/>
<meta itemprop="property" content="remove_edges_from"/>
<meta itemprop="property" content="remove_node"/>
<meta itemprop="property" content="remove_nodes_from"/>
<meta itemprop="property" content="reverse"/>
<meta itemprop="property" content="size"/>
<meta itemprop="property" content="subgraph"/>
<meta itemprop="property" content="successors"/>
<meta itemprop="property" content="to_circuit"/>
<meta itemprop="property" content="to_directed"/>
<meta itemprop="property" content="to_directed_class"/>
<meta itemprop="property" content="to_undirected"/>
<meta itemprop="property" content="to_undirected_class"/>
<meta itemprop="property" content="update"/>
</div>

# cirq.circuits.CircuitDag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A representation of a Circuit as a directed acyclic graph.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CircuitDag`, `cirq.circuits.circuit_dag.CircuitDag`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.CircuitDag(
    can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = cirq.circuits.CircuitDag.disjoint_qubits,
    incoming_graph_data: Any = None,
    device: <a href="../../cirq/devices/Device.md"><code>cirq.devices.Device</code></a> = cirq.devices.UNCONSTRAINED_DEVICE
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Nodes of the graph are instances of Unique containing each operation of a
circuit.

Edges of the graph are tuples of nodes.  Each edge specifies a required
application order between two operations.  The first must be applied before
the second.

The graph is maximalist (transitive completion).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`can_reorder`
</td>
<td>
A predicate that determines if two operations may be
reordered.  Graph edges are created for pairs of operations
where this returns False.

The default predicate allows reordering only when the operations
don't share common qubits.
</td>
</tr><tr>
<td>
`incoming_graph_data`
</td>
<td>
Data in initialize the graph.  This can be any
value supported by networkx.DiGraph() e.g. an edge list or
another graph.
</td>
</tr><tr>
<td>
`device`
</td>
<td>
Hardware that the circuit should be able to run on.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`adj`
</td>
<td>
Graph adjacency object holding the neighbors of each node.

This object is a read-only dict-like structure with node keys
and neighbor-dict values.  The neighbor-dict is keyed by neighbor
to the edge-data-dict.  So `G.adj[3][2]['color'] = 'blue'` sets
the color of the edge `(3, 2)` to `"blue"`.

Iterating over G.adj behaves like a dict. Useful idioms include
`for nbr, datadict in G.adj[n].items():`.

The neighbor information is also provided by subscripting the graph.
So `for nbr, foovalue in G[node].data('foo', default=1):` works.

For directed graphs, `G.adj` holds outgoing (successor) info.
</td>
</tr><tr>
<td>
`degree`
</td>
<td>
A DegreeView for the Graph as G.degree or G.degree().

The node degree is the number of edges adjacent to the node.
The weighted node degree is the sum of the edge weights for
edges incident to that node.

This object provides an iterator for (node, degree) as well as
lookup for the degree for a single node.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.

weight : string or None, optional (default=None)
The name of an edge attribute that holds the numerical value used
as a weight.  If None, then each edge has weight 1.
The degree is the sum of the edge weights adjacent to the node.

Returns
-------
If a single node is requested
deg : int
Degree of the node

OR if multiple nodes are requested
nd_iter : iterator
The iterator returns two-tuples of (node, degree).

See Also
--------
in_degree, out_degree

Examples
--------
>>> G = nx.DiGraph()   # or MultiDiGraph
>>> nx.add_path(G, [0, 1, 2, 3])
>>> G.degree(0) # node 0 with degree 1
1
>>> list(G.degree([0, 1, 2]))
[(0, 1), (1, 2), (2, 2)]
</td>
</tr><tr>
<td>
`edges`
</td>
<td>
An OutEdgeView of the DiGraph as G.edges or G.edges().

edges(self, nbunch=None, data=False, default=None)

The OutEdgeView provides set-like operations on the edge-tuples
as well as edge attribute lookup. When called, it also provides
an EdgeDataView object which allows control of access to edge
attributes (but does not provide set-like operations).
Hence, `G.edges[u, v]['color']` provides the value of the color
attribute for edge `(u, v)` while
`for (u, v, c) in G.edges.data('color', default='red'):`
iterates through all the edges yielding the color attribute
with default `'red'` if no color attribute exists.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.
data : string or bool, optional (default=False)
The edge attribute returned in 3-tuple (u, v, ddict[data]).
If True, return edge attribute dict in 3-tuple (u, v, ddict).
If False, return 2-tuple (u, v).
default : value, optional (default=None)
Value used for edges that don't have the requested attribute.
Only relevant if data is not True or False.

Returns
-------
edges : OutEdgeView
A view of edge attributes, usually it iterates over (u, v)
or (u, v, d) tuples of edges, but can also be used for
attribute lookup as `edges[u, v]['foo']`.

See Also
--------
in_edges, out_edges

Notes
-----
Nodes in nbunch that are not in the graph will be (quietly) ignored.
For directed graphs this returns the out-edges.

Examples
--------
>>> G = nx.DiGraph()   # or MultiDiGraph, etc
>>> nx.add_path(G, [0, 1, 2])
>>> G.add_edge(2, 3, weight=5)
>>> [e for e in G.edges]
[(0, 1), (1, 2), (2, 3)]
>>> G.edges.data()  # default data is {} (empty dict)
OutEdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
>>> G.edges.data('weight', default=1)
OutEdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
>>> G.edges([0, 2])  # only edges incident to these nodes
OutEdgeDataView([(0, 1), (2, 3)])
>>> G.edges(0)  # only edges incident to a single node (use G.adj[0]?)
OutEdgeDataView([(0, 1)])
</td>
</tr><tr>
<td>
`in_degree`
</td>
<td>
An InDegreeView for (node, in_degree) or in_degree for single node.

The node in_degree is the number of edges pointing to the node.
The weighted node degree is the sum of the edge weights for
edges incident to that node.

This object provides an iteration over (node, in_degree) as well as
lookup for the degree for a single node.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.

weight : string or None, optional (default=None)
The name of an edge attribute that holds the numerical value used
as a weight.  If None, then each edge has weight 1.
The degree is the sum of the edge weights adjacent to the node.

Returns
-------
If a single node is requested
deg : int
In-degree of the node

OR if multiple nodes are requested
nd_iter : iterator
The iterator returns two-tuples of (node, in-degree).

See Also
--------
degree, out_degree

Examples
--------
>>> G = nx.DiGraph()
>>> nx.add_path(G, [0, 1, 2, 3])
>>> G.in_degree(0) # node 0 with degree 0
0
>>> list(G.in_degree([0, 1, 2]))
[(0, 0), (1, 1), (2, 1)]
</td>
</tr><tr>
<td>
`in_edges`
</td>
<td>
An InEdgeView of the Graph as G.in_edges or G.in_edges().

in_edges(self, nbunch=None, data=False, default=None):

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.
data : string or bool, optional (default=False)
The edge attribute returned in 3-tuple (u, v, ddict[data]).
If True, return edge attribute dict in 3-tuple (u, v, ddict).
If False, return 2-tuple (u, v).
default : value, optional (default=None)
Value used for edges that don't have the requested attribute.
Only relevant if data is not True or False.

Returns
-------
in_edges : InEdgeView
A view of edge attributes, usually it iterates over (u, v)
or (u, v, d) tuples of edges, but can also be used for
attribute lookup as `edges[u, v]['foo']`.

See Also
--------
edges
</td>
</tr><tr>
<td>
`name`
</td>
<td>
String identifier of the graph.

This graph attribute appears in the attribute dict G.graph
keyed by the string `"name"`. as well as an attribute (technically
a property) `G.name`. This is entirely user controlled.
</td>
</tr><tr>
<td>
`nodes`
</td>
<td>
A NodeView of the Graph as G.nodes or G.nodes().

Can be used as `G.nodes` for data lookup and for set-like operations.
Can also be used as `G.nodes(data='color', default=None)` to return a
NodeDataView which reports specific node data but no set operations.
It presents a dict-like interface as well with `G.nodes.items()`
iterating over `(node, nodedata)` 2-tuples and `G.nodes[3]['foo']`
providing the value of the `foo` attribute for node `3`. In addition,
a view `G.nodes.data('foo')` provides a dict-like interface to the
`foo` attribute of each node. `G.nodes.data('foo', default=1)`
provides a default for nodes that do not have attribute `foo`.

Parameters
----------
data : string or bool, optional (default=False)
The node attribute returned in 2-tuple (n, ddict[data]).
If True, return entire node attribute dict as (n, ddict).
If False, return just the nodes n.

default : value, optional (default=None)
Value used for nodes that don't have the requested attribute.
Only relevant if data is not True or False.

Returns
-------
NodeView
Allows set-like operations over the nodes as well as node
attribute dict lookup and calling to get a NodeDataView.
A NodeDataView iterates over `(n, data)` and has no set operations.
A NodeView iterates over `n` and includes set operations.

When called, if data is False, an iterator over nodes.
Otherwise an iterator of 2-tuples (node, attribute value)
where the attribute is specified in `data`.
If data is True then the attribute becomes the
entire data dictionary.

Notes
-----
If your node data is not needed, it is simpler and equivalent
to use the expression ``for n in G``, or ``list(G)``.

Examples
--------
There are two simple ways of getting a list of all nodes in the graph:

```
>>> G = nx.path_graph(3)
>>> list(G.nodes)
[0, 1, 2]
>>> list(G)
[0, 1, 2]
```

To get the node data along with the nodes:

```
>>> G.add_node(1, time='5pm')
>>> G.nodes[0]['foo'] = 'bar'
>>> list(G.nodes(data=True))
[(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
>>> list(G.nodes.data())
[(0, {'foo': 'bar'}), (1, {'time': '5pm'}), (2, {})]
```

```
>>> list(G.nodes(data='foo'))
[(0, 'bar'), (1, None), (2, None)]
>>> list(G.nodes.data('foo'))
[(0, 'bar'), (1, None), (2, None)]
```

```
>>> list(G.nodes(data='time'))
[(0, None), (1, '5pm'), (2, None)]
>>> list(G.nodes.data('time'))
[(0, None), (1, '5pm'), (2, None)]
```

```
>>> list(G.nodes(data='time', default='Not Available'))
[(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
>>> list(G.nodes.data('time', default='Not Available'))
[(0, 'Not Available'), (1, '5pm'), (2, 'Not Available')]
```

If some of your nodes have an attribute and the rest are assumed
to have a default attribute value you can create a dictionary
from node/attribute pairs using the `default` keyword argument
to guarantee the value is never None::

```
>>> G = nx.Graph()
>>> G.add_node(0)
>>> G.add_node(1, weight=2)
>>> G.add_node(2, weight=3)
>>> dict(G.nodes(data='weight', default=1))
{0: 1, 1: 2, 2: 3}
```
</td>
</tr><tr>
<td>
`out_degree`
</td>
<td>
An OutDegreeView for (node, out_degree)

The node out_degree is the number of edges pointing out of the node.
The weighted node degree is the sum of the edge weights for
edges incident to that node.

This object provides an iterator over (node, out_degree) as well as
lookup for the degree for a single node.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.

weight : string or None, optional (default=None)
The name of an edge attribute that holds the numerical value used
as a weight.  If None, then each edge has weight 1.
The degree is the sum of the edge weights adjacent to the node.

Returns
-------
If a single node is requested
deg : int
Out-degree of the node

OR if multiple nodes are requested
nd_iter : iterator
The iterator returns two-tuples of (node, out-degree).

See Also
--------
degree, in_degree

Examples
--------
>>> G = nx.DiGraph()
>>> nx.add_path(G, [0, 1, 2, 3])
>>> G.out_degree(0) # node 0 with degree 1
1
>>> list(G.out_degree([0, 1, 2]))
[(0, 1), (1, 1), (2, 1)]
</td>
</tr><tr>
<td>
`out_edges`
</td>
<td>
An OutEdgeView of the DiGraph as G.edges or G.edges().

edges(self, nbunch=None, data=False, default=None)

The OutEdgeView provides set-like operations on the edge-tuples
as well as edge attribute lookup. When called, it also provides
an EdgeDataView object which allows control of access to edge
attributes (but does not provide set-like operations).
Hence, `G.edges[u, v]['color']` provides the value of the color
attribute for edge `(u, v)` while
`for (u, v, c) in G.edges.data('color', default='red'):`
iterates through all the edges yielding the color attribute
with default `'red'` if no color attribute exists.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
The view will only report edges incident to these nodes.
data : string or bool, optional (default=False)
The edge attribute returned in 3-tuple (u, v, ddict[data]).
If True, return edge attribute dict in 3-tuple (u, v, ddict).
If False, return 2-tuple (u, v).
default : value, optional (default=None)
Value used for edges that don't have the requested attribute.
Only relevant if data is not True or False.

Returns
-------
edges : OutEdgeView
A view of edge attributes, usually it iterates over (u, v)
or (u, v, d) tuples of edges, but can also be used for
attribute lookup as `edges[u, v]['foo']`.

See Also
--------
in_edges, out_edges

Notes
-----
Nodes in nbunch that are not in the graph will be (quietly) ignored.
For directed graphs this returns the out-edges.

Examples
--------
>>> G = nx.DiGraph()   # or MultiDiGraph, etc
>>> nx.add_path(G, [0, 1, 2])
>>> G.add_edge(2, 3, weight=5)
>>> [e for e in G.edges]
[(0, 1), (1, 2), (2, 3)]
>>> G.edges.data()  # default data is {} (empty dict)
OutEdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {'weight': 5})])
>>> G.edges.data('weight', default=1)
OutEdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])
>>> G.edges([0, 2])  # only edges incident to these nodes
OutEdgeDataView([(0, 1), (2, 3)])
>>> G.edges(0)  # only edges incident to a single node (use G.adj[0]?)
OutEdgeDataView([(0, 1)])
</td>
</tr><tr>
<td>
`pred`
</td>
<td>
Graph adjacency object holding the predecessors of each node.

This object is a read-only dict-like structure with node keys
and neighbor-dict values.  The neighbor-dict is keyed by neighbor
to the edge-data-dict.  So `G.pred[2][3]['color'] = 'blue'` sets
the color of the edge `(3, 2)` to `"blue"`.

Iterating over G.pred behaves like a dict. Useful idioms include
`for nbr, datadict in G.pred[n].items():`.  A data-view not provided
by dicts also exists: `for nbr, foovalue in G.pred[node].data('foo'):`
A default can be set via a `default` argument to the `data` method.
</td>
</tr><tr>
<td>
`succ`
</td>
<td>
Graph adjacency object holding the successors of each node.

This object is a read-only dict-like structure with node keys
and neighbor-dict values.  The neighbor-dict is keyed by neighbor
to the edge-data-dict.  So `G.succ[3][2]['color'] = 'blue'` sets
the color of the edge `(3, 2)` to `"blue"`.

Iterating over G.succ behaves like a dict. Useful idioms include
`for nbr, datadict in G.succ[n].items():`.  A data-view not provided
by dicts also exists: `for nbr, foovalue in G.succ[node].data('foo'):`
and a default can be set via a `default` argument to the `data` method.

The neighbor information is also provided by subscripting the graph.
So `for nbr, foovalue in G[node].data('foo', default=1):` works.

For directed graphs, `G.adj` is identical to `G.succ`.
</td>
</tr>
</table>



## Child Classes
[`class adjlist_inner_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

[`class adjlist_outer_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

[`class edge_attr_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

[`class graph_attr_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

[`class node_attr_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

[`class node_dict_factory`](../../cirq/circuits/CircuitDag/adjlist_inner_dict_factory.md)

## Methods

<h3 id="add_edge"><code>add_edge</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edge(
    u_of_edge, v_of_edge, **attr
)
</code></pre>

Add an edge between u and v.

The nodes u and v will be automatically added if they are
not already in the graph.

Edge attributes can be specified with keywords or by directly
accessing the edge's attribute dictionary. See examples below.

Parameters
----------
u, v : nodes
    Nodes can be, for example, strings or numbers.
    Nodes must be hashable (and not None) Python objects.
attr : keyword arguments, optional
    Edge data (or labels or objects) can be assigned using
    keyword arguments.

See Also
--------
add_edges_from : add a collection of edges

Notes
-----
Adding an edge that already exists updates the edge data.

Many NetworkX algorithms designed for weighted graphs use
an edge attribute (by default `weight`) to hold a numerical value.

Examples
--------
The following all add the edge e=(1, 2) to graph G:

```
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> e = (1, 2)
>>> G.add_edge(1, 2)           # explicit two-node form
>>> G.add_edge(*e)             # single edge as tuple of two nodes
>>> G.add_edges_from( [(1, 2)] ) # add edges from iterable container
```

Associate data to edges using keywords:

```
>>> G.add_edge(1, 2, weight=3)
>>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)
```

For non-string attribute keys, use subscript notation.

```
>>> G.add_edge(1, 2)
>>> G[1][2].update({0: 5})
>>> G.edges[1, 2].update({0: 5})
```

<h3 id="add_edges_from"><code>add_edges_from</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_edges_from(
    ebunch_to_add, **attr
)
</code></pre>

Add all the edges in ebunch_to_add.

Parameters
----------
ebunch_to_add : container of edges
    Each edge given in the container will be added to the
    graph. The edges must be given as 2-tuples (u, v) or
    3-tuples (u, v, d) where d is a dictionary containing edge data.
attr : keyword arguments, optional
    Edge data (or labels or objects) can be assigned using
    keyword arguments.

See Also
--------
add_edge : add a single edge
add_weighted_edges_from : convenient way to add weighted edges

Notes
-----
Adding the same edge twice has no effect but any edge data
will be updated when each duplicate edge is added.

Edge attributes specified in an ebunch take precedence over
attributes specified via keyword arguments.

Examples
--------
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_edges_from([(0, 1), (1, 2)]) # using a list of edge tuples
>>> e = zip(range(0, 3), range(1, 4))
>>> G.add_edges_from(e) # Add the path graph 0-1-2-3

Associate data to edges

```
>>> G.add_edges_from([(1, 2), (2, 3)], weight=3)
>>> G.add_edges_from([(3, 4), (1, 4)], label='WN2898')
```

<h3 id="add_node"><code>add_node</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_node(
    node_for_adding, **attr
)
</code></pre>

Add a single node `node_for_adding` and update node attributes.

Parameters
----------
node_for_adding : node
    A node can be any hashable Python object except None.
attr : keyword arguments, optional
    Set or change node attributes using key=value.

See Also
--------
add_nodes_from

Examples
--------
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_node(1)
>>> G.add_node('Hello')
>>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
>>> G.add_node(K3)
>>> G.number_of_nodes()
3

Use keywords set/change node attributes:

```
>>> G.add_node(1, size=10)
>>> G.add_node(3, weight=0.4, UTM=('13S', 382871, 3972649))
```

Notes
-----
A hashable object is one that can be used as a key in a Python
dictionary. This includes strings, numbers, tuples of strings
and numbers, etc.

On many platforms hashable items also include mutables such as
NetworkX Graphs, though one should be careful that the hash
doesn't change on mutables.

<h3 id="add_nodes_from"><code>add_nodes_from</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_nodes_from(
    nodes_for_adding, **attr
)
</code></pre>

Add multiple nodes.

Parameters
----------
nodes_for_adding : iterable container
    A container of nodes (list, dict, set, etc.).
    OR
    A container of (node, attribute dict) tuples.
    Node attributes are updated using the attribute dict.
attr : keyword arguments, optional (default= no attributes)
    Update attributes for all nodes in nodes.
    Node attributes specified in nodes as a tuple take
    precedence over attributes specified via keyword arguments.

See Also
--------
add_node

Examples
--------
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_nodes_from('Hello')
>>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])
>>> G.add_nodes_from(K3)
>>> sorted(G.nodes(), key=str)
[0, 1, 2, 'H', 'e', 'l', 'o']

Use keywords to update specific node attributes for every node.

```
>>> G.add_nodes_from([1, 2], size=10)
>>> G.add_nodes_from([3, 4], weight=0.4)
```

Use (node, attrdict) tuples to update attributes for specific nodes.

```
>>> G.add_nodes_from([(1, dict(size=11)), (2, {'color':'blue'})])
>>> G.nodes[1]['size']
11
>>> H = nx.Graph()
>>> H.add_nodes_from(G.nodes(data=True))
>>> H.nodes[1]['size']
11
```

<h3 id="add_weighted_edges_from"><code>add_weighted_edges_from</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>add_weighted_edges_from(
    ebunch_to_add, weight='weight', **attr
)
</code></pre>

Add weighted edges in `ebunch_to_add` with specified weight attr

Parameters
----------
ebunch_to_add : container of edges
    Each edge given in the list or container will be added
    to the graph. The edges must be given as 3-tuples (u, v, w)
    where w is a number.
weight : string, optional (default= 'weight')
    The attribute name for the edge weights to be added.
attr : keyword arguments, optional (default= no attributes)
    Edge attributes to add/update for all edges.

See Also
--------
add_edge : add a single edge
add_edges_from : add multiple edges

Notes
-----
Adding the same edge twice for Graph/DiGraph simply updates
the edge data. For MultiGraph/MultiDiGraph, duplicate edges
are stored.

Examples
--------
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_weighted_edges_from([(0, 1, 3.0), (1, 2, 7.5)])

<h3 id="adjacency"><code>adjacency</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adjacency()
</code></pre>

Returns an iterator over (node, adjacency dict) tuples for all nodes.

For directed graphs, only outgoing neighbors/adjacencies are included.

Returns
-------
adj_iter : iterator
   An iterator over (node, adjacency dictionary) for all nodes in
   the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> [(n, nbrdict) for n, nbrdict in G.adjacency()]
[(0, {1: {}}), (1, {0: {}, 2: {}}), (2, {1: {}, 3: {}}), (3, {2: {}})]

<h3 id="all_operations"><code>all_operations</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all_operations() -> Iterator[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>]
</code></pre>




<h3 id="all_qubits"><code>all_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>all_qubits()
</code></pre>




<h3 id="append"><code>append</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>append(
    op: "cirq.Operation"
) -> None
</code></pre>




<h3 id="clear"><code>clear</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>

Remove all nodes and edges from the graph.

This also removes the name, and all graph, node, and edge attributes.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.clear()
>>> list(G.nodes)
[]
>>> list(G.edges)
[]

<h3 id="copy"><code>copy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy(
    as_view=False
)
</code></pre>

Returns a copy of the graph.

The copy method by default returns an independent shallow copy
of the graph and attributes. That is, if an attribute is a
container, that container is shared by the original an the copy.
Use Python's `copy.deepcopy` for new containers.

If `as_view` is True then a view is returned instead of a copy.

Notes
-----
All copies reproduce the graph structure, but data attributes
may be handled in different ways. There are four types of copies
of a graph that people might want.

Deepcopy -- A "deepcopy" copies the graph structure as well as
all data attributes and any objects they might contain.
The entire graph object is new so that changes in the copy
do not affect the original object. (see Python's copy.deepcopy)

Data Reference (Shallow) -- For a shallow copy the graph structure
is copied but the edge, node and graph attribute dicts are
references to those in the original graph. This saves
time and memory but could cause confusion if you change an attribute
in one graph and it changes the attribute in the other.
NetworkX does not provide this level of shallow copy.

Independent Shallow -- This copy creates new independent attribute
dicts and then does a shallow copy of the attributes. That is, any
attributes that are containers are shared between the new graph
and the original. This is exactly what `dict.copy()` provides.
You can obtain this style copy using:

    ```
    >>> G = nx.path_graph(5)
    >>> H = G.copy()
    >>> H = G.copy(as_view=False)
    >>> H = nx.Graph(G)
    >>> H = G.__class__(G)
    ```

Fresh Data -- For fresh data, the graph structure is copied while
new empty data attribute dicts are created. The resulting graph
is independent of the original and it has no edge, node or graph
attributes. Fresh copies are not enabled. Instead use:

    ```
    >>> H = G.__class__()
    >>> H.add_nodes_from(G)
    >>> H.add_edges_from(G.edges)
    ```

View -- Inspired by dict-views, graph-views act like read-only
versions of the original graph, providing a copy of the original
structure without requiring any memory for copying the information.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/2/library/copy.html.

Parameters
----------
as_view : bool, optional (default=False)
    If True, the returned graph-view provides a read-only view
    of the original graph without actually copying any data.

Returns
-------
G : Graph
    A copy of the graph.

See Also
--------
to_directed: return a directed copy of the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> H = G.copy()

<h3 id="disjoint_qubits"><code>disjoint_qubits</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>disjoint_qubits(
    op1: "cirq.Operation",
    op2: "cirq.Operation"
) -> bool
</code></pre>

Returns true only if the operations have qubits in common.


<h3 id="edge_subgraph"><code>edge_subgraph</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>edge_subgraph(
    edges
)
</code></pre>

Returns the subgraph induced by the specified edges.

The induced subgraph contains each edge in `edges` and each
node incident to any one of those edges.

Parameters
----------
edges : iterable
    An iterable of edges in this graph.

Returns
-------
G : Graph
    An edge-induced subgraph of this graph with the same edge
    attributes.

Notes
-----
The graph, edge, and node attributes in the returned subgraph
view are references to the corresponding attributes in the original
graph. The view is read-only.

To create a full graph version of the subgraph with its own copy
of the edge or node attributes, use::

    ```
    >>> G.edge_subgraph(edges).copy()  # doctest: +SKIP
    ```

Examples
--------
>>> G = nx.path_graph(5)
>>> H = G.edge_subgraph([(0, 1), (3, 4)])
>>> list(H.nodes)
[0, 1, 3, 4]
>>> list(H.edges)
[(0, 1), (3, 4)]

<h3 id="findall_nodes_until_blocked"><code>findall_nodes_until_blocked</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>findall_nodes_until_blocked(
    is_blocker: Callable[[<a href="../../cirq/ops/Operation.md"><code>cirq.ops.Operation</code></a>], bool]
) -> Iterator[Unique[ops.Operation]]
</code></pre>

Finds all nodes before blocking ones.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`is_blocker`
</td>
<td>
The predicate that indicates whether or not an
operation is blocking.
</td>
</tr>
</table>



<h3 id="from_circuit"><code>from_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_circuit(
    circuit: <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>,
    can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = cirq.circuits.CircuitDag.disjoint_qubits
) -> "CircuitDag"
</code></pre>




<h3 id="from_ops"><code>from_ops</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>from_ops(
    *operations,
    can_reorder: Callable[['cirq.Operation', 'cirq.Operation'], bool] = cirq.circuits.CircuitDag.disjoint_qubits,
    device: <a href="../../cirq/devices/Device.md"><code>cirq.devices.Device</code></a> = cirq.devices.UNCONSTRAINED_DEVICE
) -> "CircuitDag"
</code></pre>




<h3 id="get_edge_data"><code>get_edge_data</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_edge_data(
    u, v, default=None
)
</code></pre>

Returns the attribute dictionary associated with edge (u, v).

This is identical to `G[u][v]` except the default is returned
instead of an exception if the edge doesn't exist.

Parameters
----------
u, v : nodes
default:  any Python object (default=None)
    Value to return if the edge (u, v) is not found.

Returns
-------
edge_dict : dictionary
    The edge attribute dictionary.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G[0][1]
{}

Warning: Assigning to `G[u][v]` is not permitted.
But it is safe to assign attributes `G[u][v]['foo']`

```
>>> G[0][1]['weight'] = 7
>>> G[0][1]['weight']
7
>>> G[1][0]['weight']
7
```

```
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.get_edge_data(0, 1)  # default edge data is {}
{}
>>> e = (0, 1)
>>> G.get_edge_data(*e)  # tuple form
{}
>>> G.get_edge_data('a', 'b', default=0)  # edge not in graph, return 0
0
```

<h3 id="has_edge"><code>has_edge</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>has_edge(
    u, v
)
</code></pre>

Returns True if the edge (u, v) is in the graph.

This is the same as `v in G[u]` without KeyError exceptions.

Parameters
----------
u, v : nodes
    Nodes can be, for example, strings or numbers.
    Nodes must be hashable (and not None) Python objects.

Returns
-------
edge_ind : bool
    True if edge is in the graph, False otherwise.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.has_edge(0, 1)  # using two nodes
True
>>> e = (0, 1)
>>> G.has_edge(*e)  #  e is a 2-tuple (u, v)
True
>>> e = (0, 1, {'weight':7})
>>> G.has_edge(*e[:2])  # e is a 3-tuple (u, v, data_dictionary)
True

The following syntax are equivalent:

```
>>> G.has_edge(0, 1)
True
>>> 1 in G[0]  # though this gives KeyError if 0 not in G
True
```

<h3 id="has_node"><code>has_node</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>has_node(
    n
)
</code></pre>

Returns True if the graph contains the node n.

Identical to `n in G`

Parameters
----------
n : node

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.has_node(0)
True

It is more readable and simpler to use

```
>>> 0 in G
True
```

<h3 id="has_predecessor"><code>has_predecessor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>has_predecessor(
    u, v
)
</code></pre>

Returns True if node u has predecessor v.

This is true if graph has the edge u<-v.

<h3 id="has_successor"><code>has_successor</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>has_successor(
    u, v
)
</code></pre>

Returns True if node u has successor v.

This is true if graph has the edge u->v.

<h3 id="is_directed"><code>is_directed</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_directed()
</code></pre>

Returns True if graph is directed, False otherwise.


<h3 id="is_multigraph"><code>is_multigraph</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_multigraph()
</code></pre>

Returns True if graph is a multigraph, False otherwise.


<h3 id="make_node"><code>make_node</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@staticmethod</code>
<code>make_node(
    op: "cirq.Operation"
) -> <a href="../../cirq/circuits/Unique.md"><code>cirq.circuits.Unique</code></a>
</code></pre>




<h3 id="nbunch_iter"><code>nbunch_iter</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>nbunch_iter(
    nbunch=None
)
</code></pre>

Returns an iterator over nodes contained in nbunch that are
also in the graph.

The nodes in nbunch are checked for membership in the graph
and if not are silently ignored.

Parameters
----------
nbunch : single node, container, or all nodes (default= all nodes)
    The view will only report edges incident to these nodes.

Returns
-------
niter : iterator
    An iterator over nodes in nbunch that are also in the graph.
    If nbunch is None, iterate over all nodes in the graph.

Raises
------
NetworkXError
    If nbunch is not a node or or sequence of nodes.
    If a node in nbunch is not hashable.

See Also
--------
Graph.__iter__

Notes
-----
When nbunch is an iterator, the returned iterator yields values
directly from nbunch, becoming exhausted when nbunch is exhausted.

To test whether nbunch is a single node, one can use
"if nbunch in self:", even after processing with this routine.

If nbunch is not a node or a (possibly empty) sequence/iterator
or None, a :exc:`NetworkXError` is raised.  Also, if any object in
nbunch is not hashable, a :exc:`NetworkXError` is raised.

<h3 id="neighbors"><code>neighbors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>neighbors(
    n
)
</code></pre>

Returns an iterator over successor nodes of n.

A successor of n is a node m such that there exists a directed
edge from n to m.

Parameters
----------
n : node
   A node in the graph

Raises
-------
NetworkXError
   If n is not in the graph.

See Also
--------
predecessors

Notes
-----
neighbors() and successors() are the same.

<h3 id="number_of_edges"><code>number_of_edges</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>number_of_edges(
    u=None, v=None
)
</code></pre>

Returns the number of edges between two nodes.

Parameters
----------
u, v : nodes, optional (default=all edges)
    If u and v are specified, return the number of edges between
    u and v. Otherwise return the total number of all edges.

Returns
-------
nedges : int
    The number of edges in the graph.  If nodes `u` and `v` are
    specified return the number of edges between those nodes. If
    the graph is directed, this only returns the number of edges
    from `u` to `v`.

See Also
--------
size

Examples
--------
For undirected graphs, this method counts the total number of
edges in the graph:

```
>>> G = nx.path_graph(4)
>>> G.number_of_edges()
3
```

If you specify two nodes, this counts the total number of edges
joining the two nodes:

```
>>> G.number_of_edges(0, 1)
1
```

For directed graphs, this method can count the total number of
directed edges from `u` to `v`:

```
>>> G = nx.DiGraph()
>>> G.add_edge(0, 1)
>>> G.add_edge(1, 0)
>>> G.number_of_edges(0, 1)
1
```

<h3 id="number_of_nodes"><code>number_of_nodes</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>number_of_nodes()
</code></pre>

Returns the number of nodes in the graph.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
order, __len__  which are identical

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.number_of_nodes()
3

<h3 id="order"><code>order</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>order()
</code></pre>

Returns the number of nodes in the graph.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
number_of_nodes, __len__  which are identical

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.order()
3

<h3 id="ordered_nodes"><code>ordered_nodes</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>ordered_nodes() -> Iterator[Unique[ops.Operation]]
</code></pre>




<h3 id="predecessors"><code>predecessors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>predecessors(
    n
)
</code></pre>

Returns an iterator over predecessor nodes of n.

A predecessor of n is a node m such that there exists a directed
edge from m to n.

Parameters
----------
n : node
   A node in the graph

Raises
-------
NetworkXError
   If n is not in the graph.

See Also
--------
successors

<h3 id="remove_edge"><code>remove_edge</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_edge(
    u, v
)
</code></pre>

Remove the edge between u and v.

Parameters
----------
u, v : nodes
    Remove the edge between nodes u and v.

Raises
------
NetworkXError
    If there is not an edge between u and v.

See Also
--------
remove_edges_from : remove a collection of edges

Examples
--------
>>> G = nx.Graph()   # or DiGraph, etc
>>> nx.add_path(G, [0, 1, 2, 3])
>>> G.remove_edge(0, 1)
>>> e = (1, 2)
>>> G.remove_edge(*e) # unpacks e from an edge tuple
>>> e = (2, 3, {'weight':7}) # an edge with attribute data
>>> G.remove_edge(*e[:2]) # select first part of edge tuple

<h3 id="remove_edges_from"><code>remove_edges_from</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_edges_from(
    ebunch
)
</code></pre>

Remove all edges specified in ebunch.

Parameters
----------
ebunch: list or container of edge tuples
    Each edge given in the list or container will be removed
    from the graph. The edges can be:

        - 2-tuples (u, v) edge between u and v.
        - 3-tuples (u, v, k) where k is ignored.

See Also
--------
remove_edge : remove a single edge

Notes
-----
Will fail silently if an edge in ebunch is not in the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> ebunch = [(1, 2), (2, 3)]
>>> G.remove_edges_from(ebunch)

<h3 id="remove_node"><code>remove_node</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_node(
    n
)
</code></pre>

Remove node n.

Removes the node n and all adjacent edges.
Attempting to remove a non-existent node will raise an exception.

Parameters
----------
n : node
   A node in the graph

Raises
-------
NetworkXError
   If n is not in the graph.

See Also
--------
remove_nodes_from

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> list(G.edges)
[(0, 1), (1, 2)]
>>> G.remove_node(1)
>>> list(G.edges)
[]

<h3 id="remove_nodes_from"><code>remove_nodes_from</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>remove_nodes_from(
    nodes
)
</code></pre>

Remove multiple nodes.

Parameters
----------
nodes : iterable container
    A container of nodes (list, dict, set, etc.).  If a node
    in the container is not in the graph it is silently ignored.

See Also
--------
remove_node

Examples
--------
>>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> e = list(G.nodes)
>>> e
[0, 1, 2]
>>> G.remove_nodes_from(e)
>>> list(G.nodes)
[]

<h3 id="reverse"><code>reverse</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reverse(
    copy=True
)
</code></pre>

Returns the reverse of the graph.

The reverse is a graph with the same nodes and edges
but with the directions of the edges reversed.

Parameters
----------
copy : bool optional (default=True)
    If True, return a new DiGraph holding the reversed edges.
    If False, the reverse graph is created using a view of
    the original graph.

<h3 id="size"><code>size</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>size(
    weight=None
)
</code></pre>

Returns the number of edges or total of all edge weights.

Parameters
----------
weight : string or None, optional (default=None)
    The edge attribute that holds the numerical value used
    as a weight. If None, then each edge has weight 1.

Returns
-------
size : numeric
    The number of edges or
    (if weight keyword is provided) the total weight sum.

    If weight is None, returns an int. Otherwise a float
    (or more general numeric if the weights are more general).

See Also
--------
number_of_edges

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.size()
3

```
>>> G = nx.Graph()   # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G.add_edge('a', 'b', weight=2)
>>> G.add_edge('b', 'c', weight=4)
>>> G.size()
2
>>> G.size(weight='weight')
6.0
```

<h3 id="subgraph"><code>subgraph</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>subgraph(
    nodes
)
</code></pre>

Returns a SubGraph view of the subgraph induced on `nodes`.

The induced subgraph of the graph contains the nodes in `nodes`
and the edges between those nodes.

Parameters
----------
nodes : list, iterable
    A container of nodes which will be iterated through once.

Returns
-------
G : SubGraph View
    A subgraph view of the graph. The graph structure cannot be
    changed but node/edge attributes can and are shared with the
    original graph.

Notes
-----
The graph, edge and node attributes are shared with the original graph.
Changes to the graph structure is ruled out by the view, but changes
to attributes are reflected in the original graph.

To create a subgraph with its own copy of the edge/node attributes use:
G.subgraph(nodes).copy()

For an inplace reduction of a graph to a subgraph you can remove nodes:
G.remove_nodes_from([n for n in G if n not in set(nodes)])

Subgraph views are sometimes NOT what you want. In most cases where
you want to do more than simply look at the induced edges, it makes
more sense to just create the subgraph as its own graph with code like:

::

    # Create a subgraph SG based on a (possibly multigraph) G
    SG = G.__class__()
    SG.add_nodes_from((n, G.nodes[n]) for n in largest_wcc)
    if SG.is_multigraph:
        SG.add_edges_from((n, nbr, key, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, keydict in nbrs.items() if nbr in largest_wcc
            for key, d in keydict.items())
    else:
        SG.add_edges_from((n, nbr, d)
            for n, nbrs in G.adj.items() if n in largest_wcc
            for nbr, d in nbrs.items() if nbr in largest_wcc)
    SG.graph.update(G.graph)

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> H = G.subgraph([0, 1, 2])
>>> list(H.edges)
[(0, 1), (1, 2)]

<h3 id="successors"><code>successors</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>successors(
    n
)
</code></pre>

Returns an iterator over successor nodes of n.

A successor of n is a node m such that there exists a directed
edge from n to m.

Parameters
----------
n : node
   A node in the graph

Raises
-------
NetworkXError
   If n is not in the graph.

See Also
--------
predecessors

Notes
-----
neighbors() and successors() are the same.

<h3 id="to_circuit"><code>to_circuit</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_circuit() -> <a href="../../cirq/circuits/Circuit.md"><code>cirq.circuits.Circuit</code></a>
</code></pre>




<h3 id="to_directed"><code>to_directed</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_directed(
    as_view=False
)
</code></pre>

Returns a directed representation of the graph.

Returns
-------
G : DiGraph
    A directed graph with the same name, same nodes, and with
    each edge (u, v, data) replaced by two directed edges
    (u, v, data) and (v, u, data).

Notes
-----
This returns a "deepcopy" of the edge, node, and
graph attributes which attempts to completely copy
all of the data and references.

This is in contrast to the similar D=DiGraph(G) which returns a
shallow copy of the data.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/2/library/copy.html.

Warning: If you have subclassed Graph to use dict-like objects
in the data structure, those changes do not transfer to the
DiGraph created by this method.

Examples
--------
>>> G = nx.Graph()  # or MultiGraph, etc
>>> G.add_edge(0, 1)
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1), (1, 0)]

If already directed, return a (deep) copy

```
>>> G = nx.DiGraph()  # or MultiDiGraph, etc
>>> G.add_edge(0, 1)
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1)]
```

<h3 id="to_directed_class"><code>to_directed_class</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_directed_class()
</code></pre>

Returns the class to use for empty directed copies.

If you subclass the base classes, use this to designate
what directed class to use for `to_directed()` copies.

<h3 id="to_undirected"><code>to_undirected</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_undirected(
    reciprocal=False, as_view=False
)
</code></pre>

Returns an undirected representation of the digraph.

Parameters
----------
reciprocal : bool (optional)
  If True only keep edges that appear in both directions
  in the original digraph.
as_view : bool (optional, default=False)
  If True return an undirected view of the original directed graph.

Returns
-------
G : Graph
    An undirected graph with the same name and nodes and
    with edge (u, v, data) if either (u, v, data) or (v, u, data)
    is in the digraph.  If both edges exist in digraph and
    their edge data is different, only one edge is created
    with an arbitrary choice of which edge data to use.
    You must check and correct for this manually if desired.

See Also
--------
Graph, copy, add_edge, add_edges_from

Notes
-----
If edges in both directions (u, v) and (v, u) exist in the
graph, attributes for the new undirected edge will be a combination of
the attributes of the directed edges.  The edge data is updated
in the (arbitrary) order that the edges are encountered.  For
more customized control of the edge attributes use add_edge().

This returns a "deepcopy" of the edge, node, and
graph attributes which attempts to completely copy
all of the data and references.

This is in contrast to the similar G=DiGraph(D) which returns a
shallow copy of the data.

See the Python copy module for more information on shallow
and deep copies, https://docs.python.org/2/library/copy.html.

Warning: If you have subclassed DiGraph to use dict-like objects
in the data structure, those changes do not transfer to the
Graph created by this method.

Examples
--------
>>> G = nx.path_graph(2)   # or MultiGraph, etc
>>> H = G.to_directed()
>>> list(H.edges)
[(0, 1), (1, 0)]
>>> G2 = H.to_undirected()
>>> list(G2.edges)
[(0, 1)]

<h3 id="to_undirected_class"><code>to_undirected_class</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_undirected_class()
</code></pre>

Returns the class to use for empty undirected copies.

If you subclass the base classes, use this to designate
what directed class to use for `to_directed()` copies.

<h3 id="update"><code>update</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update(
    edges=None, nodes=None
)
</code></pre>

Update the graph using nodes/edges/graphs as input.

Like dict.update, this method takes a graph as input, adding the
graph's nodes and edges to this graph. It can also take two inputs:
edges and nodes. Finally it can take either edges or nodes.
To specify only nodes the keyword `nodes` must be used.

The collections of edges and nodes are treated similarly to
the add_edges_from/add_nodes_from methods. When iterated, they
should yield 2-tuples (u, v) or 3-tuples (u, v, datadict).

Parameters
----------
edges : Graph object, collection of edges, or None
    The first parameter can be a graph or some edges. If it has
    attributes `nodes` and `edges`, then it is taken to be a
    Graph-like object and those attributes are used as collections
    of nodes and edges to be added to the graph.
    If the first parameter does not have those attributes, it is
    treated as a collection of edges and added to the graph.
    If the first argument is None, no edges are added.
nodes : collection of nodes, or None
    The second parameter is treated as a collection of nodes
    to be added to the graph unless it is None.
    If `edges is None` and `nodes is None` an exception is raised.
    If the first parameter is a Graph, then `nodes` is ignored.

Examples
--------
>>> G = nx.path_graph(5)
>>> G.update(nx.complete_graph(range(4,10)))
>>> from itertools import combinations
>>> edges = ((u, v, {'power': u * v})
...          for u, v in combinations(range(10, 20), 2)
...          if u * v < 225)
>>> nodes = [1000]  # for singleton, use a container
>>> G.update(edges, nodes)

Notes
-----
It you want to update the graph using an adjacency structure
it is straightforward to obtain the edges/nodes from adjacency.
The following examples provide common cases, your adjacency may
be slightly different and require tweaks of these examples.

```
>>> # dict-of-set/list/tuple
>>> adj = {1: {2, 3}, 2: {1, 3}, 3: {1, 2}}
>>> e = [(u, v) for u, nbrs in adj.items() for v in  nbrs]
>>> G.update(edges=e, nodes=adj)
```

```
>>> DG = nx.DiGraph()
>>> # dict-of-dict-of-attribute
>>> adj = {1: {2: 1.3, 3: 0.7}, 2: {1: 1.4}, 3: {1: 0.7}}
>>> e = [(u, v, {'weight': d}) for u, nbrs in adj.items()
...      for v, d in nbrs.items()]
>>> DG.update(edges=e, nodes=adj)
```

```
>>> # dict-of-dict-of-dict
>>> adj = {1: {2: {'weight': 1.3}, 3: {'color': 0.7, 'weight':1.2}}}
>>> e = [(u, v, {'weight': d}) for u, nbrs in adj.items()
...      for v, d in nbrs.items()]
>>> DG.update(edges=e, nodes=adj)
```

```
>>> # predecessor adjacency (dict-of-set)
>>> pred = {1: {2, 3}, 2: {3}, 3: {3}}
>>> e = [(v, u) for u, nbrs in pred.items() for v in nbrs]
```

```
>>> # MultiGraph dict-of-dict-of-dict-of-attribute
>>> MDG = nx.MultiDiGraph()
>>> adj = {1: {2: {0: {'weight': 1.3}, 1: {'weight': 1.2}}},
...        3: {2: {0: {'weight': 0.7}}}}
>>> e = [(u, v, ekey, d) for u, nbrs in adj.items()
...      for v, keydict in nbrs.items()
...      for ekey, d in keydict.items()]
>>> MDG.update(edges=e)
```

See Also
--------
add_edges_from: add multiple edges to a graph
add_nodes_from: add multiple nodes to a graph

<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    n
)
</code></pre>

Returns True if n is a node, False otherwise. Use: 'n in G'.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> 1 in G
True

<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    n
)
</code></pre>

Returns a dict of neighbors of node n.  Use: 'G[n]'.

Parameters
----------
n : node
   A node in the graph.

Returns
-------
adj_dict : dictionary
   The adjacency dictionary for nodes connected to n.

Notes
-----
G[n] is the same as G.adj[n] and similar to G.neighbors(n)
(which is an iterator over G.adj[n])

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> G[0]
AtlasView({1: {}})

<h3 id="__iter__"><code>__iter__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Iterate over the nodes. Use: 'for n in G'.

Returns
-------
niter : iterator
    An iterator over all nodes in the graph.

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> [n for n in G]
[0, 1, 2, 3]
>>> list(G)
[0, 1, 2, 3]

<h3 id="__len__"><code>__len__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Returns the number of nodes in the graph. Use: 'len(G)'.

Returns
-------
nnodes : int
    The number of nodes in the graph.

See Also
--------
number_of_nodes, order  which are identical

Examples
--------
>>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc
>>> len(G)
4

<h3 id="__ne__"><code>__ne__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/circuits/circuit_dag.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    other
)
</code></pre>

Return self!=value.




