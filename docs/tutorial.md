# Tutorial

In this tutorial we will go from knowing nothing about Cirq to creating a
quantum variational algorithm. Note that this tutorial isn't a quantum
computing 101 tutorial, we assume familiarity of quantum computing at about
the level of the textbook "Quantum Computation and Quantum Information" by
Nielsen and Chuang.  

To begin, please follow the instructions for [installing Cirq](install.md)

### Variational quantum algorithms

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

~[Energy](resources/EnergyDef.gif)

### Circuits, Moments, OP_TREEs

