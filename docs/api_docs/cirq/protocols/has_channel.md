<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.has_channel" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.has_channel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/channel.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns whether the value has a channel representation.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.has_channel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.has_channel(
    val: Any
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a `_has_channel_` method and its result is not
NotImplemented, that result is returned. Otherwise, if `val` has a
`_has_mixture_` method and its result is not NotImplemented, that
result is returned. Otherwise if `val` has a `_has_unitary_` method
and its results is not NotImplemented, that result is returned.
Otherwise, if the value has a _channel_ method return if that
has a non-default value. Returns False if none of these functions
exists.
</td>
</tr>

</table>

