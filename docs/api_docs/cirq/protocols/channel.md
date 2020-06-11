<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.channel" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.channel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/channel.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns a list of matrices describing the channel for the given value.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.channel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.channel(
    val: Any,
    default: Any = RaiseTypeErrorIfNotProvided
) -> Union[Tuple[np.ndarray], Sequence[TDefault]]
</code></pre>



<!-- Placeholder for "Used in" -->

These matrices are the terms in the operator sum representation of
a quantum channel. If the returned matrices are {A_0,A_1,..., A_{r-1}},
then this describes the channel:
    \rho \rightarrow \sum_{k=0}^{r-1} A_0 \rho A_0^\dagger
These matrices are required to satisfy the trace preserving condition
    \sum_{k=0}^{r-1} A_i^\dagger A_i = I
where I is the identity matrix. The matrices A_i are sometimes called
Krauss or noise operators.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`val`
</td>
<td>
The value to describe by a channel.
</td>
</tr><tr>
<td>
`default`
</td>
<td>
Determines the fallback behavior when `val` doesn't have
a channel. If `default` is not set, a TypeError is raised. If
default is set to a value, that value is returned.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
If `val` has a `_channel_` method and its result is not NotImplemented,
that result is returned. Otherwise, if `val` has a `_mixture_` method
and its results is not NotImplement a tuple made up of channel
corresponding to that mixture being a probabilistic mixture of unitaries
is returned.  Otherwise, if `val` has a `_unitary_` method and
its result is not NotImplemented a tuple made up of that result is
returned. Otherwise, if a default value was specified, the default
value is returned.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`
</td>
<td>
`val` doesn't have a _channel_ or _unitary_ method (or that
method returned NotImplemented) and also no default value was
specified.
</td>
</tr>
</table>

