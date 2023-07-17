# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gates for preparing coefficient states.

In section III.D. of the [Linear T paper](https://arxiv.org/abs/1805.03662) the authors introduce
a technique for initializing a state with $L$ unique coefficients (provided by a classical
database) with a number of T gates scaling as 4L + O(log(1/eps)) where eps is the
largest absolute error that one can tolerate in the prepared amplitudes.
"""

from typing import List
from numpy.typing import NDArray

import attr
import cirq
import numpy as np
from cirq._compat import cached_property
from cirq_ft import infra, linalg
from cirq_ft.algos import (
    arithmetic_gates,
    prepare_uniform_superposition,
    qrom,
    select_and_prepare,
    swap_network,
)


@cirq.value_equality()
@attr.frozen
class StatePreparationAliasSampling(select_and_prepare.PrepareOracle):
    r"""Initialize a state with $L$ unique coefficients using coherent alias sampling.

    In particular, we take the zero state to:

    $$
    \sum_{\ell=0}^{L-1} \sqrt{p_\ell} |\ell\rangle |\mathrm{temp}_\ell\rangle
    $$

    where the probabilities $p_\ell$ are $\mu$-bit binary approximations to the true values and
    where the temporary register must be treated with care, see the details in Section III.D. of
    the reference.

    The preparation is equivalent to [classical alias sampling]
    (https://en.wikipedia.org/wiki/Alias_method): we sample `l` with probability `p[l]` by first
    selecting `l` uniformly at random and then returning it with probability `keep[l] / 2**mu`;
    otherwise returning `alt[l]`.

    Registers:
        selection: The input/output register $|\ell\rangle$ of size lg(L) where the desired
            coefficient state is prepared.
        temp: Work space comprised of sub registers:
            - sigma: A mu-sized register containing uniform probabilities for comparison against
                `keep`.
            - alt: A lg(L)-sized register of alternate indices
            - keep: a mu-sized register of probabilities of keeping the initially sampled index.
            - one bit for the result of the comparison.

    This gate corresponds to the following operations:
     - UNIFORM_L on the selection register
     - H^mu on the sigma register
     - QROM addressed by the selection register into the alt and keep registers.
     - LessThanEqualGate comparing the keep and sigma registers.
     - Coherent swap between the selection register and alt register if the comparison
       returns True.

    Total space will be (2 * log(L) + 2 mu + 1) work qubits + log(L) ancillas for QROM.
    The 1 ancilla in work qubits is for the `LessThanEqualGate` followed by coherent swap.

    References:
            [Encoding Electronic Spectra in Quantum Circuits with Linear T Complexity]
        (https://arxiv.org/abs/1805.03662).
        Babbush et. al. (2018). Section III.D. and Figure 11.
    """
    selection_registers: infra.SelectionRegisters
    alt: NDArray[np.int_]
    keep: NDArray[np.int_]
    mu: int

    @classmethod
    def from_lcu_probs(
        cls, lcu_probabilities: List[float], *, probability_epsilon: float = 1.0e-5
    ) -> 'StatePreparationAliasSampling':
        """Factory to construct the state preparation gate for a given set of LCU coefficients.

        Args:
            lcu_probabilities: The LCU coefficients.
            probability_epsilon: The desired accuracy to represent each probability
                (which sets mu size and keep/alt integers).
                See `cirq_ft.linalg.lcu_util.preprocess_lcu_coefficients_for_reversible_sampling`
                for more information.
        """
        alt, keep, mu = linalg.preprocess_lcu_coefficients_for_reversible_sampling(
            lcu_coefficients=lcu_probabilities, epsilon=probability_epsilon
        )
        N = len(lcu_probabilities)
        return StatePreparationAliasSampling(
            selection_registers=infra.SelectionRegisters(
                [infra.SelectionRegister('selection', (N - 1).bit_length(), N)]
            ),
            alt=np.array(alt),
            keep=np.array(keep),
            mu=mu,
        )

    @cached_property
    def sigma_mu_bitsize(self) -> int:
        return self.mu

    @cached_property
    def alternates_bitsize(self) -> int:
        return self.selection_registers.total_bits()

    @cached_property
    def keep_bitsize(self) -> int:
        return self.mu

    @cached_property
    def selection_bitsize(self) -> int:
        return self.selection_registers.total_bits()

    @cached_property
    def junk_registers(self) -> infra.Registers:
        return infra.Registers.build(
            sigma_mu=self.sigma_mu_bitsize,
            alt=self.alternates_bitsize,
            keep=self.keep_bitsize,
            less_than_equal=1,
        )

    def _value_equality_values_(self):
        return (
            self.selection_registers,
            tuple(self.alt.ravel()),
            tuple(self.keep.ravel()),
            self.mu,
        )

    def __repr__(self) -> str:
        alt_repr = cirq._compat.proper_repr(self.alt)
        keep_repr = cirq._compat.proper_repr(self.keep)
        return (
            f'cirq_ft.StatePreparationAliasSampling('
            f'{self.selection_registers}, '
            f'{alt_repr}, '
            f'{keep_repr}, '
            f'{self.mu})'
        )

    def decompose_from_registers(
        self,
        *,
        context: cirq.DecompositionContext,
        **quregs: NDArray[cirq.Qid],  # type:ignore[type-var]
    ) -> cirq.OP_TREE:
        selection, less_than_equal = quregs['selection'], quregs['less_than_equal']
        sigma_mu, alt, keep = quregs['sigma_mu'], quregs['alt'], quregs['keep']
        N = self.selection_registers[0].iteration_length
        yield prepare_uniform_superposition.PrepareUniformSuperposition(N).on(*selection)
        yield cirq.H.on_each(*sigma_mu)
        qrom_gate = qrom.QROM(
            [self.alt, self.keep],
            (self.selection_bitsize,),
            (self.alternates_bitsize, self.keep_bitsize),
        )
        yield qrom_gate.on_registers(selection=selection, target0=alt, target1=keep)
        yield arithmetic_gates.LessThanEqualGate(self.mu, self.mu).on(
            *keep, *sigma_mu, *less_than_equal
        )
        yield swap_network.MultiTargetCSwap.make_on(
            control=less_than_equal, target_x=alt, target_y=selection
        )
