<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.value.big_endian_int_to_bits" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.value.big_endian_int_to_bits

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/value/digits.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the big-endian bits of an integer.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.big_endian_int_to_bits`, `cirq.value.digits.big_endian_int_to_bits`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.value.big_endian_int_to_bits(
    val: int,
    *,
    bit_count: int
) -> List[int]
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The integer to get bits from. This integer is permitted to be
larger than `2**bit_count` (in which case the high bits of the
result are dropped) or to be negative (in which case the bits come
from the 2s complement signed representation).
</td>
</tr><tr>
<td>
`bit_count`
</td>
<td>
The number of desired bits in the result.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The bits.
</td>
</tr>

</table>



#### Examples:

>>> cirq.big_endian_int_to_bits(19, bit_count=8)
[0, 0, 0, 1, 0, 0, 1, 1]

```
>>> cirq.big_endian_int_to_bits(19, bit_count=4)
[0, 0, 1, 1]
```

```
>>> cirq.big_endian_int_to_bits(-3, bit_count=4)
[1, 1, 0, 1]
```
