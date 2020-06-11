<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.testing.assert_json_roundtrip_works" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.testing.assert_json_roundtrip_works

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/testing/json.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Tests that the given object can serialized and de-serialized

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.testing.json.assert_json_roundtrip_works`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.testing.assert_json_roundtrip_works(
    obj, text_should_be=None, resolvers=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`obj`
</td>
<td>
The object to test round-tripping for.
</td>
</tr><tr>
<td>
`text_should_be`
</td>
<td>
An optional argument to assert the JSON serialized
output.
</td>
</tr><tr>
<td>
`resolvers`
</td>
<td>
Any resolvers if testing those other than the default.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
The given object can not be round-tripped according to
the given arguments.
</td>
</tr>
</table>

