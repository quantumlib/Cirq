<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.linalg.scatter_plot_normalized_kak_interaction_coefficients" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.linalg.scatter_plot_normalized_kak_interaction_coefficients

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/linalg/decompositions.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Plots the interaction coefficients of many two-qubit operations.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.linalg.decompositions.scatter_plot_normalized_kak_interaction_coefficients`, `cirq.scatter_plot_normalized_kak_interaction_coefficients`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.linalg.scatter_plot_normalized_kak_interaction_coefficients(
    interactions: Iterable[Union[np.ndarray, 'cirq.SupportsUnitary', 'KakDecomposition']],
    *,
    include_frame: bool = True,
    ax: Optional[plt.Axes] = None,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Plots:

A point for the (x, y, z) normalized interaction coefficients of
each interaction from the given interactions. The (x, y, z) coordinates
are normalized so that the maximum value is at 1 instead of at pi/4.

If `include_frame` is set to True, then a black wireframe outline of the
canonicalized normalized KAK coefficient space. The space is defined by
the following two constraints:

    0 <= abs(z) <= y <= x <= 1
    if x = 1 then z >= 0

The wireframe includes lines along the surface of the space at z=0.

The space is a prism with the identity at the origin, a crease along
y=z=0 leading to the CZ/CNOT at x=1 and a vertical triangular face that
contains the iswap at x=y=1,z=0 and the swap at x=y=z=1:

                         (x=1,y=1,z=0)
                     swap___iswap___swap (x=1,y=1,z=+-1)
                       _/\    |    /
                     _/   \   |   /
                   _/      \  |  /
                 _/         \ | /
               _/            \|/
(x=0,y=0,z=0) I---------------CZ (x=1,y=0,z=0)



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`interactions`
</td>
<td>
An iterable of two qubit unitary interactions. Each
interaction can be specified as a raw 4x4 unitary matrix, or an
object with a 4x4 unitary matrix according to <a href="../../cirq/protocols/unitary.md"><code>cirq.unitary</code></a> (
(e.g. <a href="../../cirq/ops/CZ.md"><code>cirq.CZ</code></a> or a <a href="../../cirq/linalg/KakDecomposition.md"><code>cirq.KakDecomposition</code></a> or a <a href="../../cirq/circuits/Circuit.md"><code>cirq.Circuit</code></a>
over two qubits).
</td>
</tr><tr>
<td>
`include_frame`
</td>
<td>
Determines whether or not to draw the kak space
wireframe. Defaults to `True`.
</td>
</tr><tr>
<td>
`ax`
</td>
<td>
A matplotlib 3d axes object to plot into. If not specified, a new
figure is created, plotted, and shown.
</td>
</tr><tr>
<td>
`kwargs`
</td>
<td>
Arguments forwarded into the call to `scatter` that plots the
points. Working arguments include color `c='blue'`, scale `s=2`,
labelling `label="theta=pi/4"`, etc. For reference see the
`matplotlib.pyplot.scatter` documentation:
https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.scatter.html
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The matplotlib 3d axes object that was plotted into.
</td>
</tr>

</table>



#### Examples:

>>> ax = None
>>> for y in np.linspace(0, 0.5, 4):
...     a, b = cirq.LineQubit.range(2)
...     circuits = [
...         cirq.Circuit(
...             cirq.CZ(a, b)**0.5,
...             cirq.X(a)**y, cirq.X(b)**x,
...             cirq.CZ(a, b)**0.5,
...             cirq.X(a)**x, cirq.X(b)**y,
...             cirq.CZ(a, b) ** 0.5,
...         )
...         for x in np.linspace(0, 1, 25)
...     ]
...     ax = cirq.scatter_plot_normalized_kak_interaction_coefficients(
...         circuits,
...         include_frame=ax is None,
...         ax=ax,
...         s=1,
...         label=f'y={y:0.2f}')
>>> _ = ax.legend()
>>> import matplotlib.pyplot as plt
>>> plt.show()
