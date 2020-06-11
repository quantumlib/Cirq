<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.SupportsMeasurementKey" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.protocols.SupportsMeasurementKey

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/measurement_key_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object that is a measurement and has a measurement key or keys.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SupportsMeasurementKey`, `cirq.protocols.measurement_key_protocol.SupportsMeasurementKey`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.SupportsMeasurementKey(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Measurement keys are used in referencing the results of a measurement.

Users are free to implement either `_measurement_key_` returning one string
or `_measurement_keys_` returning an iterable of strings.

Note: Measurements, in contrast to general quantum channels, are
distinguished by the recording of the quantum operation that occurred.
That is a general quantum channel may enact the evolution
    \rho \rightarrow \sum_k A_k \rho A_k^\dagger
where as a measurement enacts the evolution
    \rho \rightarrow A_k \rho A_k^\dagger
conditional on the measurement outcome being k.

