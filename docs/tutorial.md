# Tutorial

In this tutorial we will go from knowing nothing about Cirq to creating a
quantum variational algorithm. Note that this tutorial isn't a quantum
computing 101 tutorial, we assume familiarity of quantum computing at about
the level of the textbook "Quantum Computation and Quantum Information" by
Nielsen and Chuang.  

To begin, please follow the instructions for [installing Cirq](install.md)

### Background: Variational quantum algorithms

The (variational method)[https://en.wikipedia.org/wiki/Variational_method_(quantum_mechanics)]
in quantum theory is a way of finding low energy states of a system.
The rough idea of this method is that one defines a trial wave function
(sometimes called an ansatz) as a function of some parameter, and then
one finds the values of these parameters that minimize the expectation
value of the energy with respect to these parameters. This minimized
anstaz is then an approximation to the lowest energy eigenstate, and 
the expectation values serves as an upper bound on the energy of the
ground state.

Recently it has been realized that quantum computers can mimic the
classical technique and that a quantum computer does so with certain
advantages.  In particular, when one applies the classical variational
method to a system of `n` qubits, an exponential number (in `n`) number
of complex numbers are necessary to generically represent the 
wave function of the system.  However with a quantum computer one
can directly produce this state using a parameterized quantum
circuit, and then by repeated measurements estimate the expectation
value of the energy.

This idea has led to a class of algorithms known as variational quantum
algorithms. Indeed this approach is not just limited to finding low
energy eigenstates, but minimizing any objective function that can
be expressed as a quantum observable.In general one does not know
whether these variational algorithms will succeed or not, and exploring
this class of algorithms is a key part of research for noisy intermediate
scale quantum computers.

The classical problem we will focus on is the 2D +/- Ising model with
transverse field (ISING).  This problem is NP-complete so it is highly unlikely
that quantum computers will be able to efficiently solve it.  Yet this
type of problem is illustrative of the general class of problems that
Cirq is designed to tackle.

Consider the energy function

![Energy](resources/EnergyDef.gif)

where here each s<sub>i</sub>, J<sub>i,j</sub>, and h<sub>i</sub> is either 
+1 or -1.  Here each index i is associated with a bit on a square lattice,
and the <i,j> notation means sums over neighboring bits on this lattice.
The problem we would like to solve is, given J<sub>i,j</sub>, and h<sub>i</sub>,
find an assignment of s<sub>i</sub>s that minimize E.  

What does a variational quantum algorithm work for this? One approach is
to consider `n` qubits and associate them with each of the bits in the 
classical problem.  This maps the classical problem onto the quantum problem
of minimizing the expectation value of the observable

![Hamiltonian](resources/HamiltonianDef.gif)

Then one defines a set of parameterized quantum circuits, i.e. a 
quantum circuit where the gates (or more general quantum operations)
are parameterized by some values.  This produces an ansatz state

![State definition](resources/StateDef.gif)

where p<sub>i</sub> are the parameters that produce this state
(here we assume a pure state, but mixed states are of course
possible).

The variational algorithm this proceeds by
1. Prepare the ansatz state.
2. Make a measurement which samples from terms in H.
Note that one cannot always measure H directly (without
the use of quantum phase estimation), so one often relies
on the linearity of expectation values to measure parts of
H and then by repeating the steps above one can estimate 
the expectation value of these terms which can then be added.
In particular one always needs to repeat 1 and 2 to make
an estimate of the expectation value.  An important
point is that one needs to repeat this measurement multiple
times and that one will almost always have an error in this
estimate.  How many measurements to make is beyond the 
scope of this tutorial, but Cirq can help investigate
this tradeoff. 

### Circuits and Moments

To build the above variational quantum algorithm using Cirq, 
one begins by building the appropriate [circuit](circuit.md).
