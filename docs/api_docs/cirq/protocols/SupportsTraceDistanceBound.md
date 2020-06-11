<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.SupportsTraceDistanceBound" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.protocols.SupportsTraceDistanceBound

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/trace_distance_bound.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An effect with known bounds on how easy it is to detect.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SupportsTraceDistanceBound`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.SupportsTraceDistanceBound(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Used when deciding whether or not an operation is negligible. For example,
the trace distance between the states before and after a Z**0.00000001
operation is very close to 0, so it would typically be considered
negligible.

