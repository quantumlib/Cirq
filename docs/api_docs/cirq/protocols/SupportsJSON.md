<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.SupportsJSON" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__new__"/>
</div>

# cirq.protocols.SupportsJSON

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



An object that can be turned into JSON dictionaries.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.SupportsJSON`, `cirq.protocols.json_serialization.SupportsJSON`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.SupportsJSON(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

The magic method _json_dict_ must return a trivially json-serializable
type or other objects that support the SupportsJSON protocol.

During deserialization, a class must be able to be resolved (see
the docstring for `read_json`) and must be able to be (re-)constructed
from the serialized parameters. If the type defines a classmethod
`_from_json_dict_`, that will be called. Otherwise, the `cirq_type` key
will be popped from the dictionary and used as kwargs to the type's
constructor.

