<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.trace_distance_from_angle_list" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.trace_distance_from_angle_list

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/trace_distance_bound.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Given a list of arguments of the eigenvalues of a unitary matrix,

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.trace_distance_from_angle_list`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.trace_distance_from_angle_list(
    angle_list: Sequence[float]
) -> float
</code></pre>



<!-- Placeholder for "Used in" -->
calculates the trace distance bound of the unitary effect.

The maximum provided angle should not exceed the minimum provided angle
by more than 2π.