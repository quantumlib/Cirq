<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.google.SerializingArg" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="required"/>
</div>

# cirq.google.SerializingArg

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/google/op_serializer.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Specification of the arguments for a Gate and its serialization.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.google.op_serializer.SerializingArg`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.google.SerializingArg(
    serialized_name, serialized_type, op_getter, required=True
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`serialized_name`
</td>
<td>
The name of the argument when it is serialized.
</td>
</tr><tr>
<td>
`serialized_type`
</td>
<td>
The type of the argument when it is serialized.
</td>
</tr><tr>
<td>
`op_getter`
</td>
<td>
The name of the property or attribute for getting the
value of this argument from a gate, or a function that takes a
operation and returns this value. The later can be used to supply
a value of the serialized arg by supplying a lambda that
returns this value (i.e. `lambda x: default_value`)
</td>
</tr><tr>
<td>
`required`
</td>
<td>
Whether this argument is a required argument for the
serialized form.
</td>
</tr>
</table>



## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>






## Class Variables

* `required = True` <a id="required"></a>
