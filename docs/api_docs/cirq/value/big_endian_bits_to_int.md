<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.big_endian_bits_to_int" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.big_endian_bits_to_int

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/digits.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the big-endian integer specified by the given bits.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.big_endian_bits_to_int`, `cirq.value.digits.big_endian_bits_to_int`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.big_endian_bits_to_int(
    bits: Iterable[Any]
) -> int
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`bits`
</td>
<td>
Descending bits of the integer, with the 1s bit at the end.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The integer.
</td>
</tr>

</table>



#### Examples:


```
>>> cirq.big_endian_bits_to_int([0, 1])
1
```

```
>>> cirq.big_endian_bits_to_int([1, 0])
2
```

```
>>> cirq.big_endian_bits_to_int([0, 1, 0])
2
```

```
>>> cirq.big_endian_bits_to_int([1, 0, 0, 1, 0])
18
```
