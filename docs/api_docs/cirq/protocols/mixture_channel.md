<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="cirq.protocols.mixture_channel" />
<meta itemprop="path" content="Stable" />
</div>

# cirq.protocols.mixture_channel

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api" align="left">

<td>
  <a target="_blank" href="https://github.com/quantumlib/cirq/tree/master/cirq/protocols/mixture_protocol.py">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



THIS FUNCTION IS DEPRECATED.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`cirq.mixture_channel`, `cirq.protocols.mixture_protocol.mixture_channel`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cirq.protocols.mixture_channel(
    val: Any,
    default: Any = cirq.protocols.mixture_protocol.RaiseTypeErrorIfNotProvided
) -> Sequence[Tuple[float, np.ndarray]]
</code></pre>



<!-- Placeholder for "Used in" -->

IT WILL BE REMOVED IN `cirq v0.10.0`.

Use "cirq.mixture" instead.

Return a sequence of tuples representing a probabilistic unitary.

    A mixture is described by an iterable of tuples of the form

        (probability of unitary, unitary as numpy array)

    The probability components of the tuples must sum to 1.0 and be
    non-negative.

    Args:
        val: The value to decompose into a mixture of unitaries.
        default: A default value if val does not support mixture.

    Returns:
        An iterable of tuples of size 2. The first element of the tuple is a
        probability (between 0 and 1) and the second is the object that occurs
        with that probability in the mixture. The probabilities will sum to 1.0.
    