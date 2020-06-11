<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.experiments.GridInteractionLayer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="col_offset"/>
<meta itemprop="property" content="stagger"/>
<meta itemprop="property" content="vertical"/>
</div>

# cirq.experiments.GridInteractionLayer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/random_quantum_circuit_generation.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A layer of aligned or staggered two-qubit interactions on a grid.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.experiments.random_quantum_circuit_generation.GridInteractionLayer`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.experiments.GridInteractionLayer(
    col_offset=0, vertical=False, stagger=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Layers of this type have two different basic structures,
aligned:

*-* *-* *-*
*-* *-* *-*
*-* *-* *-*
*-* *-* *-*
*-* *-* *-*
*-* *-* *-*

and staggered:

*-* *-* *-*
* *-* *-* *
*-* *-* *-*
* *-* *-* *
*-* *-* *-*
* *-* *-* *

Other variants are obtained by offsetting these lattices to the right by
some number of columns, and/or transposing into the vertical orientation.
There are a total of 4 aligned and 4 staggered variants.

The 2x2 unit cells for the aligned and staggered versions of this layer
are, respectively:

*-*
*-*

and

*-*
* *-

with left/top qubits at (0, 0) and (1, 0) in the aligned case, or
(0, 0) and (1, 1) in the staggered case. Other variants have the same unit
cells after transposing and offsetting.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`col_offset`
</td>
<td>
Number of columns by which to shift the basic lattice.
</td>
</tr><tr>
<td>
`vertical`
</td>
<td>
Whether gates should be oriented vertically rather than
horizontally.
</td>
</tr><tr>
<td>
`stagger`
</td>
<td>
Whether to stagger gates in neighboring rows.
</td>
</tr>
</table>



## Methods

<h3 id="__contains__"><code>__contains__</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/experiments/random_quantum_circuit_generation.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    pair
) -> bool
</code></pre>

Checks whether a pair is in this layer.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






## Class Variables

* `col_offset = 0` <a id="col_offset"></a>
* `stagger = False` <a id="stagger"></a>
* `vertical = False` <a id="vertical"></a>
