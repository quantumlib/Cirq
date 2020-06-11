<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.json_serialization.CirqEncoder" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="default"/>
<meta itemprop="property" content="encode"/>
<meta itemprop="property" content="iterencode"/>
<meta itemprop="property" content="item_separator"/>
<meta itemprop="property" content="key_separator"/>
</div>

# cirq.protocols.json_serialization.CirqEncoder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Extend json.JSONEncoder to support Cirq objects.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.json_serialization.CirqEncoder(
    *, skipkeys=False, ensure_ascii=True, check_circular=True, allow_nan=True,
    sort_keys=False, indent=None, separators=None, default=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This supports custom serialization. For details, see the documentation
for the SupportsJSON protocol.

In addition to serializing objects that implement the SupportsJSON
protocol, this encoder deals with common, basic types:

 - Python complex numbers get saved as a dictionary keyed by 'real'
   and 'imag'.
 - Numpy ndarrays are converted to lists to use the json module's
   built-in support for lists.
 - Preliminary support for Sympy objects. Currently only sympy.Symbol.
   See https://github.com/quantumlib/Cirq/issues/2014

## Methods

<h3 id="default"><code>default</code></h3>

<a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/json_serialization.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>default(
    o
)
</code></pre>

Implement this method in a subclass such that it returns
a serializable object for ``o``, or calls the base implementation
(to raise a ``TypeError``).

For example, to support arbitrary iterators, you could
implement default like this::

    def default(self, o):
        try:
            iterable = iter(o)
        except TypeError:
            pass
        else:
            return list(iterable)
        # Let the base class default method raise the TypeError
        return JSONEncoder.default(self, o)

<h3 id="encode"><code>encode</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>encode(
    o
)
</code></pre>

Return a JSON string representation of a Python data structure.

```
>>> from json.encoder import JSONEncoder
>>> JSONEncoder().encode({"foo": ["bar", "baz"]})
'{"foo": ["bar", "baz"]}'
```

<h3 id="iterencode"><code>iterencode</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>iterencode(
    o, _one_shot=False
)
</code></pre>

Encode the given object and yield each string
representation as available.

For example::

    for chunk in JSONEncoder().iterencode(bigobject):
        mysocket.write(chunk)



## Class Variables

* `item_separator = ', '` <a id="item_separator"></a>
* `key_separator = ': '` <a id="key_separator"></a>
