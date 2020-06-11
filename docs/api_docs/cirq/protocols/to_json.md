<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.to_json" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.to_json

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write a JSON file containing a representation of obj.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.protocols.json_serialization.to_json`, `cirq.to_json`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.to_json(
    obj: Any,
    file_or_fn: Union[None, IO, pathlib.Path, str] = None,
    *,
    indent: int = 2
) -> Optional[str]
</code></pre>



<!-- Placeholder for "Used in" -->

The object may be a cirq object or have data members that are cirq
objects which implement the SupportsJSON protocol.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`obj`
</td>
<td>
An object which can be serialized to a JSON representation.
</td>
</tr><tr>
<td>
`file_or_fn`
</td>
<td>
A filename (if a string or `pathlib.Path`) to write to, or
an IO object (such as a file or buffer) to write to, or `None` to
indicate that the method should return the JSON text as its result.
Defaults to `None`.
</td>
</tr><tr>
<td>
`indent`
</td>
<td>
Pretty-print the resulting file with this indent level.
Passed to json.dump.
</td>
</tr><tr>
<td>
`cls`
</td>
<td>
Passed to json.dump; the default value of CirqEncoder
enables the serialization of Cirq objects which implement
the SupportsJSON protocol. To support serialization of 3rd
party classes, prefer adding the _json_dict_ magic method
to your classes rather than overriding this default.
</td>
</tr>
</table>

