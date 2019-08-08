# Copyright 2019 The Cirq Developers
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

"""An efficient simulator for Clifford circuits.

Allowed operations include: 
	- X,Y,Z,H,S,CNOT,CZ
	- measurements in the computational basis

The quantum state is specified in two forms:
    1. In terms of stabilizer generators. These are a set of n Pauli operators
    {S_1,S_2,...,S_n} such that S_i |psi> = |psi>.

    This implementation is based on Aaronson and Gottesman, 2004 (arXiv:quant-ph/0406196).

    2. In the CH-form defined by Bravyi et al, 2018 (arXiv:1808.00128). This representation keeps 
    track of overall phase and enables access to wavefunction amplitudes. 
"""

import numpy as np
import cirq
from cirq.sim import simulator
from cirq import circuits, study, ops, protocols
from typing import Dict, List, Iterator, Union, Any
import collections

class CliffordSimulator(simulator.SimulatesSamples,simulator.SimulatesIntermediateState):
    """An efficient simulator for Clifford circuits."""
    def __init__(self):
        self.init = True
    
    def _base_iterator(
            self,
            circuit: circuits.Circuit,
            qubit_order: ops.QubitOrderOrList,
            initial_state: int
    ) -> Iterator:
        """Iterator over CliffordSimulatorStepResult from Moments of a Circuit

        Args:
            circuit: The circuit to simulate.
            qubit_order: Determines the canonical ordering of the qubits. This
                is often used in specifying the initial state, i.e. the
                ordering of the computational basis states.
            initial_state: The initial state for the simulation.


        Yields:
            CliffordStepResult from simulating a Moment of the Circuit.
        """
        qubits = ops.QubitOrder.as_qubit_order(qubit_order).order_for(
                circuit.all_qubits())
        
        num_qubits = len(qubits)
        qubit_map = {q: i for i, q in enumerate(qubits)}
        
        if len(circuit) == 0:
            yield CliffordSimulatorStepResult(measurements={},
                state=CliffordState(qubit_map,initial_state=initial_state))
        else:
            state = CliffordState(qubit_map,initial_state=initial_state)
            
            for moment in circuit:
                measurements = collections.defaultdict(
                        list)
                
                for op in moment:
                    if protocols.has_unitary(op):
                        state.apply_unitary(op)
                    elif protocols.is_measurement(op):
                        key = protocols.measurement_key(op)
                        measurements[key].extend(state.perform_measurement(op.qubits))
                        
                yield CliffordSimulatorStepResult(measurements=measurements,state=state)
    
    def _simulator_iterator(
            self,
            circuit: circuits.Circuit,
            param_resolver: study.ParamResolver,
            qubit_order: ops.QubitOrderOrList,
            initial_state: int,
    ) -> Iterator:
        """See definition in `cirq.SimulatesIntermediateState`.

        Args:
            inital_state: An integer specifying the inital state in the computational basis.
        """
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        actual_initial_state = 0 if initial_state is None else initial_state
        
        return self._base_iterator(resolved_circuit,
                                   qubit_order,
                                   actual_initial_state)
    
    def _create_simulator_trial_result(self,
        params: study.ParamResolver,
        measurements: Dict[str, np.ndarray],
        final_simulator_state):
        
        return CliffordTrialResult(
            # params=params,
            measurements=measurements,
            final_simulator_state=final_simulator_state)
    
    def _run(
        self,
        circuit: circuits.Circuit,
        param_resolver: study.ParamResolver,
        repetitions: int) -> Dict[str, List[np.ndarray]]:
        
        param_resolver = param_resolver or study.ParamResolver({})
        resolved_circuit = protocols.resolve_parameters(circuit, param_resolver)
        
        measurements = {}  # type: Dict[str, List[np.ndarray]]
        for _ in range(repetitions):
            all_step_results = self._base_iterator(
                    circuit,
                    qubit_order=ops.QubitOrder.DEFAULT,
                    initial_state=0)

            for step_result in all_step_results:
                for k, v in step_result.measurements.items():
                    if not k in measurements:
                        measurements[k] = []
                    measurements[k].append(np.array(v, dtype=bool))
                    
        return {k: np.array(v) for k, v in measurements.items()}

class CliffordTrialResult(simulator.SimulationTrialResult):
    def __init__(self,
            measurements: Dict[str, np.ndarray],
            final_simulator_state: 'CliffordState') -> None:

        super().__init__(params=None,
                         measurements=measurements,
                         final_simulator_state=final_simulator_state)

        self.final_state = final_simulator_state
        
    def __str__(self):
        samples = super().__str__()
        final = self._final_simulator_state
        
        return 'measurements: {}\noutput state: {}'.format(samples, final)

    def __repr__(self):
        return super().__repr__()

class CliffordSimulatorStepResult(simulator.StepResult):
    """A `StepResult` that includes `StateVectorMixin` methods."""

    def __init__(self, state, measurements):
        """Results of a step of the simulator.
        Attributes:
            state: A CliffordState
            measurements: A dictionary from measurement gate key to measurement
                results, ordered by the qubits that the measurement operates on.
            qubit_map: A map from the Qubits in the Circuit to the the index
                of this qubit for a canonical ordering. This canonical ordering
                is used to define the state vector (see the state_vector()
                method).
        """
        self.measurements = measurements
        # self.qubit_map = qubit_map
        self.state = state
    
    def __str__(self):
        def bitstring(vals):
            return ''.join('1' if v else '0' for v in vals)

        results = sorted(
            [(key, bitstring(val)) for key, val in self.measurements.items()])
        
        if len(results) == 0:
            measurements = ''
        else:
            measurements =  ' '.join(
                ['{}={}'.format(key, val) for key, val in results]) + '\n'
        
        final = self.state
        
        return '{}{}'.format(measurements, final)
    
    def _simulator_state(self):
        return self.state
    
    def sample(self, qubits: List[ops.Qid],
               repetitions: int = 1) -> np.ndarray:
        
        measurements = []
        
        for _ in range(repetitions):
            measurements.append(self.state.perform_measurement(qubits,collapse_wavefunction=False))
        
        return np.array(measurements,dtype=bool)

class CliffordState():
    """A state of the Clifford simulation. 

    The state is stored using two complementary representations: Anderson's tableaux form
    and Bravyi's CH-form. The tableaux keeps track of the stabilizer operations, while the
    CH-form allows access to the full wavefunction (including phase).

    Gates and measurements are applied to each representation in O(n^2) time. 
    """
    def __init__(self,qubit_map,initial_state=0):
        self.qubit_map = qubit_map
        self.n = len(qubit_map)

        self.tableau = CliffordTableau(self.n,initial_state)
        self.CH_form = CH_Form(self.n,initial_state)
            
    def copy(self):
        state = CliffordState(self.qubit_map)
        state.tableau = self.tableau.copy()
        state.CH_form = self.CH_form.copy()

        return state
            
    def __repr__(self):
        return self.CH_form.__repr__()

    def to_numpy(self):
        return self.CH_form.to_numpy()

    # def print_stabilizers(self):
    #     print(self.tableau.__str__())

    def stabilizers(self):
        return self.tableau.stabilizers()

    def wave_function(self):
        return self.CH_form.wave_function()
            
    def apply_unitary(self, op: ops.Operation):
        if op.gate == cirq.CNOT:
            self.tableau._CNOT(self.qubit_map[op.qubits[0]],self.qubit_map[op.qubits[1]]) 
            self.CH_form._CNOT(self.qubit_map[op.qubits[0]],self.qubit_map[op.qubits[1]]) 
        elif op.gate == cirq.CZ:
            self.tableau._CZ(self.qubit_map[op.qubits[0]],self.qubit_map[op.qubits[1]]) 
            self.CH_form._CZ(self.qubit_map[op.qubits[0]],self.qubit_map[op.qubits[1]]) 
        elif op.gate == cirq.Z:
            self.tableau._Z(self.qubit_map[op.qubits[0]])
            self.CH_form._Z(self.qubit_map[op.qubits[0]])
        elif op.gate == cirq.X:
            self.tableau._X(self.qubit_map[op.qubits[0]])
            self.CH_form._X(self.qubit_map[op.qubits[0]])
        elif op.gate == cirq.Y:
            self.tableau._Y(self.qubit_map[op.qubits[0]])
            self.CH_form._Y(self.qubit_map[op.qubits[0]])
        elif op.gate == cirq.S:
            self.tableau._S(self.qubit_map[op.qubits[0]])
            self.CH_form._S(self.qubit_map[op.qubits[0]])
        elif op.gate == cirq.H:
            self.tableau._H(self.qubit_map[op.qubits[0]])
            self.CH_form._H(self.qubit_map[op.qubits[0]])
        else:
            raise ValueError('%s cannot be run with Clifford simulator' % str(op.gate))
            
    def perform_measurement(self, qubits: List[ops.Qid], collapse_wavefunction = True):
        results = []
        
        if collapse_wavefunction:
            state = self
        else:
            state = self.copy()
        
        for qubit in qubits:
            result = state.tableau._measure(self.qubit_map[qubit])
            state.CH_form.project_Z(self.qubit_map[qubit],result)
            results.append(result)
        
        return results

class CliffordTableau():
    """ Tableau representation of a stabilizer state (based on Aaronson and Gottesman 2006). 

    The tableau stores the stabilizer generators of the state using three binary arrays:
    xs, zs, and rs. 

    Each row of the arrays represents a Pauli string, P, that is 
    an eigenoperator of the wavefunction with eigenvalue one: P|psi> = |psi>. 
    """
    def __init__(self,num_qubits,initial_state=0):
        self.n = num_qubits

        self.rs = np.zeros(2*self.n+1,dtype=bool)

        def bits(s):
            while s > 0:
                yield s & 1
                s >>= 1

        for (i,val) in enumerate(bits(initial_state)):
            self.rs[2*self.n-i-1] = bool(val)
        
        self.xs = np.zeros((2*self.n+1,self.n),dtype=bool)
        self.zs = np.zeros((2*self.n+1,self.n),dtype=bool)
        
        for i in range(self.n):
            self.xs[i,i] = True
            self.zs[self.n+i,i] = True
            
    def copy(self):
        state = CliffordTableau(self.n)
        state.rs = self.rs.copy()
        state.xs = self.xs.copy()
        state.zs = self.zs.copy()
        
        return state
            
    def __repr__(self):
        return "stabilizers: [{}]".format(", ".join(self.stabilizers()))
            
    def __str__(self):    
        string = ""
        
        for i in range(self.n,2*self.n):
            string += "- " if self.rs[i] else "+ "

            for k in range(0,self.n):
                if self.xs[i,k] & (not self.zs[i,k]):
                    string += "X "
                elif (not self.xs[i,k]) & self.zs[i,k]:
                    string += "Z "
                elif self.xs[i,k] & self.zs[i,k]:
                    string += "Y "
                else:
                    string += "I "
                
            if i < 2*self.n - 1:
                string += "\n"
            
        return string
        
    def _str_full_(self):
        string = ""
        
        string += "stable" + " "*max(self.n*2-3,1)
        string += "| destable\n"
        string += "-"*max(7,self.n*2+3) + "+" + "-"*max(10,self.n*2+4) + "\n"

        for j in range(self.n):
            for i in [j+self.n,j]:
                string += "- " if self.rs[i] else "+ "

                for k in range(0,self.n):
                    if self.xs[i,k] & (not self.zs[i,k]):
                        string += "X%d" % k
                    elif (not self.xs[i,k]) & self.zs[i,k]:
                        string += "Z%d" % k
                    elif self.xs[i,k] & self.zs[i,k]:
                        string += "Y%d" % k
                    else:
                        string += "  "
                        
                if i == j+self.n:
                    string += " "*max(0,4-self.n*2)+" | "
                
            string += "\n"
        
        return string

    def _CZ(self,q,r):
        self._H(r)
        self._CNOT(q,r)
        self._H(r)

    def _X(self,q):
        self.rs[:] ^= self.zs[:,q]
        
    def _Y(self,q):
        self.rs[:] ^= self.xs[:,q] | self.zs[:,q]
        
    def _Z(self,q):
        self.rs[:] ^= self.xs[:,q]
        
    def _S(self,q):
        self.rs[:] ^= (self.xs[:,q] & self.zs[:,q])
        self.zs[:,q] ^= self.xs[:,q]
            
    def _H(self,q):
        (self.xs[:,q],self.zs[:,q]) = (self.zs[:,q].copy(),self.xs[:,q].copy())
        self.rs[:] ^= (self.xs[:,q] & self.zs[:,q])
            
    def _CNOT(self,q1,q2):
        self.rs[:] ^= self.xs[:,q1] & self.zs[:,q2] & (~(self.xs[:,q2] ^ self.zs[:,q1]))
        self.xs[:,q2] ^= self.xs[:,q1]
        self.zs[:,q1] ^= self.zs[:,q2]
            
    def _rowsum(self,q1,q2):
        ''' Implements the "rowsum" routine defined by Aaronson and Gottesman.
        This multiplies the stabilizer in row q1 by the stabilizer in row q2. '''
        def g(x1,z1,x2,z2):
            if not x1 and not z1:
                return 0
            elif x1 and z1:
                return int(z2)-int(x2)
            elif x1 and not z1:
                return int(z2)*(2*int(x2)-1)
            else:
                return int(x2)*(1-2*int(z2))
            
        r = 2*int(self.rs[q1])+2*int(self.rs[q2])
        for j in range(self.n):
            r += g(self.xs[q2,j],self.zs[q2,j],self.xs[q1,j],self.zs[q1,j])
        
        r %= 4
        
        self.rs[q1] = bool(r) 
        
        self.xs[q1,:] ^= self.xs[q2,:]
        self.zs[q1,:] ^= self.zs[q2,:]
    
    def stabilizers(self):
        ''' Returns the stabilizer generators of the state. These
        are n operators {S_1,S_2,...,S_n} such thate S_i |psi> = |psi> '''
        stabilizers = [] 
        
        for i in range(self.n,2*self.n):
            stabilizer = "-" if self.rs[i] else ""
            
            for k in range(self.n):
                if self.xs[i,k] & (not self.zs[i,k]):
                    stabilizer += "X%d" % k
                elif (not self.xs[i,k]) & self.zs[i,k]:
                    stabilizer += "Z%d" % k
                elif self.xs[i,k] & self.zs[i,k]:
                    stabilizer += "Y%d" % k
            
            stabilizers.append(stabilizer)
            
        return stabilizers
                
    def _measure(self,q):
        ''' Performs a projective measurement on the q'th qubit. 

        Returns: the result (0 or 1) of the measurement.
        '''
        is_commuting = True
        for i in range(self.n,2*self.n):
            if self.xs[i,q]:
                p = i
                is_commuting = False
                break
                
                
        if is_commuting:
            self.xs[2*self.n,:] = False
            self.zs[2*self.n,:] = False
            self.rs[2*self.n] = False
            
            for i in range(self.n):
                if self.xs[i,q]:
                    self._rowsum(2*self.n,self.n+i)
            return int(self.rs[2*self.n])
        
        else:
            for i in range(2*self.n):
                if i != p and self.xs[i,q]:
                    self._rowsum(i,p)

            self.xs[p-self.n,:] = self.xs[p,:]
            self.zs[p-self.n,:] = self.zs[p,:]
            self.rs[p-self.n] = self.rs[p]

            self.xs[p,:] = False
            self.zs[p,:] = False

            self.zs[p,q] = True

            self.rs[p] = bool(np.random.randint(2))

            return int(self.rs[p])


class CH_Form():
    '''
    A representation of stabilizer states using the CH form,

        |psi> = omega U_C U_H |s>

    This representation keeps track of overall phase.

    See Bravyi et al, 2016 for details.
    '''
    def __init__(self,num_qubits,initial_state=0):
        self.n = num_qubits
        
        # The state is represented by a set of binary matrices and vectors.
        # See Section IVa of Bravyi et al
        self.G = np.eye(self.n,dtype=bool)
        
        self.F = np.eye(self.n,dtype=bool)
        self.M = np.zeros((self.n,self.n),dtype=bool)
        self.gamma = np.zeros(self.n,dtype=int)
        
        self.v = np.zeros(self.n,dtype=bool)
        self.s = np.zeros(self.n,dtype=bool)
        
        self.omega = 1

        def bits(s):
            while s > 0:
                yield s & 1
                s >>= 1

        # Apply X for every non-zero element of initial_state
        for (i,val) in enumerate(bits(initial_state)):
            if val:
                self._X(self.n-i-1)

    def copy(self):
        copy = CH_Form(self.n)

        copy.G = self.G.copy()
        copy.F = self.F.copy()
        copy.M = self.M.copy()
        copy.gamma = self.gamma.copy()
        copy.v = self.v.copy()
        copy.s = self.s.copy()
        copy.omega = self.omega

        return copy
    
    def __repr__(self):
        ''' Return the wavefunction representation of the state. '''
        return cirq.dirac_notation(self.to_numpy())  
        
    def _repr_full(self):
        ''' Return the CH form representation of the state. '''
        string = ""
        
        string += "omega: {:.2f}\n".format(self.omega)
        string += "G:\n{}\n".format(np.array(self.G,dtype=int))
        string += "F:\n{}\n".format(np.array(self.F,dtype=int))
        string += "M:\n{}\n".format(np.array(self.M,dtype=int))
        string += "gamma: {}\n".format(np.array(self.gamma,dtype=int))
        string += "v: {}\n".format(np.array(self.v,dtype=int))
        string += "s: {}\n".format(np.array(self.s,dtype=int))
        
        return string
    
    def _int2arr(self,x):
        ''' Helper function that returns the bitstring representaiton of x '''
        arr = np.zeros(self.n,dtype=bool)
        
        i = self.n-1
        while x > 0:
            if x & 1:
                arr[i] = True
            i -= 1
            x >>= 1
        
        return arr
        
    def inner_product(self,x):
        ''' Returns the amplitude of x'th element of the wavefunction, i.e. <x|psi> '''
        if type(x) == int:
            x = self._int2arr(x)
            
        mu = sum(x * self.gamma)
        
        u = np.zeros(self.n,dtype=bool)
        for p in range(self.n):
            if x[p]:
                u ^= self.F[p,:]
                mu += 2*(sum(self.M[p,:] & u) % 2)
                
        return self.omega*2**(-sum(self.v)/2)*1j**mu*(-1)**sum(self.v & u & self.s)*np.all(self.v | (u==self.s))

    def wave_function(self):
        wf = np.zeros(2**self.n,dtype=complex)
        
        for x in range(2**self.n):
            wf[x] = self.inner_product(x)
            
        return wf
    
    def _S(self,q,right=False):
        if right:
            self.M[:,q] ^= self.F[:,q]
            self.gamma[:] = (self.gamma[:] - self.F[:,q]) % 4
        else:
            self.M[q,:] ^= self.G[q,:]
            self.gamma[q] = (self.gamma[q] - 1) % 4

    def _Z(self,q):
        self._S(q)
        self._S(q)

    def _X(self,q):
        self._H(q)
        self._Z(q)
        self._H(q)

    def _Y(self,q):
        self._Z(q)
        self._X(q)
        self.omega *= 1j
            
    def _CZ(self,q,r,right=False):
        if right:
            self.M[:,q] ^= self.F[:,r] 
            self.M[:,r] ^= self.F[:,q] 
            self.gamma[:] = (self.gamma[:] + 2*self.F[:,q]*self.F[:,r]) % 4
        else:
            self.M[q,:] ^= self.G[r,:] 
            self.M[r,:] ^= self.G[q,:] 
            
    def _CNOT(self,q,r,right=False):
        if right:
            self.G[:,q] ^= self.G[:,r] 
            self.F[:,r] ^= self.F[:,q] 
            self.M[:,q] ^= self.M[:,r] 
        else:
            self.gamma[q] = (self.gamma[q] + self.gamma[r] + 2*(sum(self.M[q,:]&self.F[r,:])%2)) % 4
            self.G[r,:] ^= self.G[q,:] 
            self.F[q,:] ^= self.F[r,:] 
            self.M[q,:] ^= self.M[r,:] 
            
    def _H(self,p):
        t = self.s ^ (self.G[p,:] & self.v)
        u = self.s ^ (self.F[p,:] & (~self.v)) ^ (self.M[p,:] & self.v)
        
        alpha = sum(self.G[p,:] & (~self.v) & self.s) % 2
        beta = sum(self.M[p,:] & (~self.v) & self.s) 
        beta += sum(self.F[p,:] & self.v & self.M[p,:])
        beta += sum(self.F[p,:] & self.v & self.s)
        beta %= 2
        
        delta = (self.gamma[p] + 2*(alpha+beta)) % 4
        
        self._update_sum(t,u,delta=delta,alpha=alpha)
    
    def _update_sum(self,t,u,delta=0,alpha=0):
        ''' Implements the transformation (Proposition 4 in Bravyi et al)

                i^alpha U_H (|t> + i^delta |u>) = omega W_C W_H |s'>
        '''
        if np.all(t == u):
            self.s = t
            self.omega *= 1/np.sqrt(2)*(-1)**alpha*(1+1j**delta)
        else:
            set0 = np.where((~self.v) & (t ^ u))[0]
            set1 = np.where(self.v & (t ^ u))[0]
            
            # implement Vc
            if len(set0) > 0:
                q = set0[0]
                for i in set0:
                    if i != q:
                        self._CNOT(q,i,right=True)
                for i in set1:
                    self._CZ(q,i,right=True)
            elif len(set1) > 0:
                q = set1[0]
                for i in set1:
                    if i != q:
                        self._CNOT(i,q,right=True)
                
            e = np.zeros(self.n,dtype=bool)
            e[q] = True
            
            if t[q]:
                y = u ^ e
                z = u
            else:
                y = t
                z = t ^ e
                
            (omega,a,b,c) = self._H_decompose(self.v[q],y[q],z[q],delta)
            
            self.s = y
            self.s[q] = c
            self.omega *= (-1)**alpha*omega
            
            if a:
                self._S(q,right=True)
            self.v[q] ^= b ^ self.v[q]

    def _H_decompose(self,v,y,z,delta):
        """ Determines the transformation 

                H^v (|y> + i^delta |z>) = omega S^a H^b |c>

        where the state represents a single qubit. 

        Input: v,y,z are boolean; delta is an integer (mod 4)
        Outputs: a,b,c are boolean; omega is a complex number
        
        Precondition: y != z """
        if y == z:
            raise ValueError('|y> is equal to |z>')

        if not v:
            omega = (1j)**(delta*int(y))

            delta2 = ((-1)**y * delta) % 4
            c = bool((delta2 >> 1))
            a = bool(delta2 & 1)
            b = True
        else:
            if not (delta & 1):
                a = False
                b = False
                c = bool(delta >> 1)
                omega = (-1)**(c & y)
            else:
                omega = 1/np.sqrt(2)*(1+1j**delta)
                b = True
                a = True
                c = not ((delta >> 1) ^ y)

        return omega,a,b,c

    def to_numpy(self):
        arr = np.zeros(2**self.n,dtype=complex)

        for x in range(len(arr)):
            arr[x] = self.inner_product(x)

        return arr 
    
    def project_Z(self,q,z):
        ''' Applies a Z projector on the q'th qubit. 

        Returns: a normalized state with Z_q |psi> = z |psi>
        ''' 
        t = self.s.copy()
        u = (self.G[q,:] & self.v) ^ self.s
        delta = (2*sum((self.G[q,:] & (~self.v)) & self.s) + 2*z) % 4
        
        if np.all(t == u):
            self.omega /= np.sqrt(2)
            
        self._update_sum(t,u,delta=delta)