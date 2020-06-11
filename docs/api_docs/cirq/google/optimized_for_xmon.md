<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.optimized_for_xmon" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.google.optimized_for_xmon

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/optimizers/optimize_for_xmon.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>





<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.optimizers.optimize_for_xmon.optimized_for_xmon`, `cirq.google.optimizers.optimized_for_xmon`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.optimized_for_xmon(
    circuit: "cirq.Circuit",
    new_device: Optional['cirq.google.XmonDevice'] = None,
    qubit_map: Callable[['cirq.Qid'], <a href="../../cirq/devices/GridQubit.md"><code>cirq.devices.GridQubit</code></a>] = (lambda e: cast(devices.GridQubit, e)),
    allow_partial_czs: bool = False
) -> "cirq.Circuit"
</code></pre>



<!-- Placeholder for "Used in" -->
