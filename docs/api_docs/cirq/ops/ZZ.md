<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.ops.ZZ" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.ops.ZZ

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



The Z-parity gate, possibly raised to a power.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.ZZ`, `cirq.ops.parity_gates.ZZ`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.ops.ZZ(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The ZZ**t gate implements the following unitary:

    (Z⊗Z)^t = [1 . . .]
              [. w . .]
              [. . w .]
              [. . . 1]

    where w = e^{i \pi t} and '.' means '0'.