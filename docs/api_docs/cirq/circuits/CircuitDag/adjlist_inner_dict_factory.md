<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.circuits.CircuitDag.adjlist_inner_dict_factory" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="__ne__"/>
<meta itemprop="property" content="__new__"/>
<meta itemprop="property" content="clear"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="fromkeys"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="items"/>
<meta itemprop="property" content="keys"/>
<meta itemprop="property" content="pop"/>
<meta itemprop="property" content="popitem"/>
<meta itemprop="property" content="setdefault"/>
<meta itemprop="property" content="update"/>
<meta itemprop="property" content="values"/>
</div>

# cirq.circuits.CircuitDag.adjlist_inner_dict_factory

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

</table>



dict() -> new empty dictionary

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.CircuitDag.adjlist_inner_dict_factory`, `cirq.CircuitDag.adjlist_outer_dict_factory`, `cirq.CircuitDag.edge_attr_dict_factory`, `cirq.CircuitDag.graph_attr_dict_factory`, `cirq.CircuitDag.node_attr_dict_factory`, `cirq.CircuitDag.node_dict_factory`, `cirq.circuits.CircuitDag.adjlist_outer_dict_factory`, `cirq.circuits.CircuitDag.edge_attr_dict_factory`, `cirq.circuits.CircuitDag.graph_attr_dict_factory`, `cirq.circuits.CircuitDag.node_attr_dict_factory`, `cirq.circuits.CircuitDag.node_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.adjlist_inner_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.adjlist_outer_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.edge_attr_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.graph_attr_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.node_attr_dict_factory`, `cirq.circuits.circuit_dag.CircuitDag.node_dict_factory`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.circuits.CircuitDag.adjlist_inner_dict_factory(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->
dict(mapping) -> new dictionary initialized from a mapping object's
    (key, value) pairs
dict(iterable) -> new dictionary initialized as if via:
    d = {}
    for k, v in iterable:
        d[k] = v
dict(**kwargs) -> new dictionary initialized with the name=value pairs
    in the keyword argument list.  For example:  dict(one=1, two=2)

## Methods

<h3 id="clear"><code>clear</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>clear()
</code></pre>

D.clear() -> None.  Remove all items from D.


<h3 id="copy"><code>copy</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>copy()
</code></pre>

D.copy() -> a shallow copy of D


<h3 id="fromkeys"><code>fromkeys</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>fromkeys(
    iterable, value, /
)
</code></pre>

Create a new dictionary with keys from iterable and values set to value.


<h3 id="get"><code>get</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get(
    key, default, /
)
</code></pre>

Return the value for key if key is in the dictionary, else default.


<h3 id="items"><code>items</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>items()
</code></pre>

D.items() -> a set-like object providing a view on D's items


<h3 id="keys"><code>keys</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>keys()
</code></pre>

D.keys() -> a set-like object providing a view on D's keys


<h3 id="pop"><code>pop</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>pop()
</code></pre>

D.pop(k[,d]) -> v, remove specified key and return the corresponding value.
If key is not found, d is returned if given, otherwise KeyError is raised

<h3 id="popitem"><code>popitem</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>popitem()
</code></pre>

D.popitem() -> (k, v), remove and return some (key, value) pair as a
2-tuple; but raise KeyError if D is empty.

<h3 id="setdefault"><code>setdefault</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>setdefault(
    key, default, /
)
</code></pre>

Insert key with a value of default if key is not in the dictionary.

Return the value for key if key is in the dictionary, else default.

<h3 id="update"><code>update</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update()
</code></pre>

D.update([E, ]**F) -> None.  Update D from dict/iterable E and F.
If E is present and has a .keys() method, then does:  for k in E: D[k] = E[k]
If E is present and lacks a .keys() method, then does:  for k, v in E: D[k] = v
In either case, this is followed by: for k in F:  D[k] = F[k]

<h3 id="values"><code>values</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>values()
</code></pre>

D.values() -> an object providing a view on D's values


<h3 id="__contains__"><code>__contains__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__contains__(
    key, /
)
</code></pre>

True if the dictionary has the specified key, else False.


<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    value, /
)
</code></pre>

Return self==value.


<h3 id="__ge__"><code>__ge__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    value, /
)
</code></pre>

Return self>=value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__()
</code></pre>

x.__getitem__(y) <==> x[y]


<h3 id="__gt__"><code>__gt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    value, /
)
</code></pre>

Return self>value.


<h3 id="__iter__"><code>__iter__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Implement iter(self).


<h3 id="__le__"><code>__le__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    value, /
)
</code></pre>

Return self<=value.


<h3 id="__len__"><code>__len__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Return len(self).


<h3 id="__lt__"><code>__lt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    value, /
)
</code></pre>

Return self<value.


<h3 id="__ne__"><code>__ne__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ne__(
    value, /
)
</code></pre>

Return self!=value.




