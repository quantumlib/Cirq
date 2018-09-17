# This is a version of the N = 2 qubit noisty circuit evolution. The purpose is to introduce
# into Cirq density matrix evoluton with Kraus operators. 
# Implementation by Hrant Gharibyan, Quantum AI, Google Inc. 
# The code is build to serve as a template for appripriate edits in Cirq 
# to support noisy evolution. 

import numpy as np
import scipy.linalg as lg 
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy.optimize import newton_krylov




# Useful gates we will use in the circuit simulations 
# Single qubit gates 
X = np.array([[0, 1], [1, 0]]); 
Y = np.array([[0, -1j], [1j, 0]]);
Z = np.array([[1, 0], [0, -1]]);
I2 = np.array([[1, 0], [0, 1]]);
S = np.array([[1, 0], [0, 1j]]);
H = np.array([[1, 1], [1, -1]])/np.sqrt(2);
T = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]]);

#Two qubit gates
CZ = np.array(np.matrix('1,0,0,0 ; 0,1,0,0 ; 0,0,1,0 ; 0,0,0,-1'));
SWAP  = np.array(np.matrix('1,0,0,0 ; 0,0,1,0 ; 0,1,0,0 ; 0,0,0,1'));
iSWAP  = np.array(np.matrix('1,0,0,0 ; 0,0,1j,0 ; 0,1j,0,0 ; 0,0,0,1'));

# Full gateset on two qubits (G1 \otimes G2  and  iSWAP)
gates=np.zeros([4,4,10], dtype=complex);
gates[:,:,0] = np.kron(lg.sqrtm(X), lg.sqrtm(X)); gates[:,:,1] = np.kron(lg.sqrtm(X),lg.sqrtm(Y));
gates[:,:,2] = np.kron(lg.sqrtm(X), T); gates[:,:,3] = np.kron(lg.sqrtm(Y), lg.sqrtm(X));
gates[:,:,4] = np.kron(lg.sqrtm(Y), lg.sqrtm(Y)); gates[:,:,5] = np.kron(lg.sqrtm(Y), T);
gates[:,:,6] = np.kron(T, lg.sqrtm(X)); gates[:,:,7] = np.kron(T, lg.sqrtm(Y));
gates[:,:,8] = np.kron(T, T);
gates[:,:,9] = iSWAP;

# Pauli basis for 2 qubits 
pauli=np.zeros([4,4,16], dtype=complex);
pauli[:,:,0] = 0.5*np.kron(I2, I2); pauli[:,:,1] = 0.5*np.kron(X, I2); 
pauli[:,:,2] = 0.5*np.kron(Y, I2); pauli[:,:,3] = 0.5*np.kron(Z, I2); 
pauli[:,:,4] = 0.5*np.kron(I2, X); pauli[:,:,5] = 0.5*np.kron(X, X); 
pauli[:,:,6] = 0.5*np.kron(Y, X); pauli[:,:,7] = 0.5*np.kron(Z, X); 
pauli[:,:,8] = 0.5*np.kron(I2, Y); pauli[:,:,9] = 0.5*np.kron(X, Y); 
pauli[:,:,10] = 0.5*np.kron(Y, Y); pauli[:,:,11] = 0.5*np.kron(Z, Y); 
pauli[:,:,12] = 0.5*np.kron(I2, Z); pauli[:,:,13] = 0.5*np.kron(X, Z); 
pauli[:,:,14] = 0.5*np.kron(Y, Z); pauli[:,:,15] = 0.5*np.kron(Z, Z); 





#######################################################################################
"""MODULE 1: Lease Mean Square Method """
"""This modelsimulated noisy random circuit for each depth d 
    with the depolarizing channel assumption."""
#######################################################################################

def channel(den_matrix, kraus_ops):

    '''
    Takes as input:
        den_matrix: density matrix 
        kraus_ops: array of Kraus operators on two qubits
        
    Output:
        den_out: density matrix after channel was applied 
    '''

    dim = len(den_matrix[:, 0]);
    den_out = np.zeros([dim, dim], dtype=complex);
    for i in range(len(kraus_ops[0,0,:])):
        den_out = den_out+kraus_ops[:,:,i] @ den_matrix @ kraus_ops[:,:,i].conj().T;
    return den_out



def dpchannel(den_matrix, p_dep):

    '''
    Takes as input:
        den_matrix: initial density matrix 
        p_dep: depolarizing channel parameter 
        
    Output:
        den_out: density matrix after depolariing channel was applied 
    '''

    dim = len(den_matrix[:, 0]);
    den_out = (1-p_dep)*den_matrix + (p_dep/dim)*np.eye(dim);
    
    return den_out


def func(x, a, b):
        '''
    Takes as input:
        x: variable x for the data 
        a and b: parameters of linear fit 
        
    Output:
       y: outputs a*x+b  
    '''
        return a*x + b

def iswap(w):

    '''
    Takes as input:
        w: the angular deviation from pi/2 angel 
       
        
    Output:
        M_out: iswap operator with error w from pi/2 anger 
    '''
    M_out = np.array([[1,0,0,0],[0, np.cos(np.pi/2+w), 1j*np.sin(np.pi/2+w), 0], [0, 1j*np.sin(np.pi/2+w),np.cos(np.pi/2+w),0],[0,0,0,1]]);
    return M_out
         

def frequency(x):
    '''
    Takes as input:
        x: list of elements in range(4)
        
    Output:
        f: frequency of each element  
    '''
    uni = np.unique(x) 
    f = np.zeros([4]);
    for k in uni:
        f[k] = np.bincount(x)[k];
    f=f/sum(f);
    return f

def crossentropy(p, q):
    '''
    Takes as input:
        p and q: probability distributions  
        
    Output:
        xE: computes the cross entropy between them
    '''
    dim=len(p);
    xE=0;

    for i in range(dim):
        xE=xE-p[i]*np.log(q[i]);
    return xE


def aveFidelity(kraus_ops):
    '''
    Takes as input:
        kraus_ops: set of Kraus operators for given noisy channel
       
    Output:
        Fave: computes average fidelity using Nielsen's formula 
    '''
    dim=4; # dimension of two qubit system 
    alpha = 0; # initialize alpha param.
    for i in range(len(kraus_ops[1,1,:])):
        alpha = alpha + abs(np.trace(kraus_ops[:,:,i]))**2;
    alpha=alpha/dim;
    Fave = (alpha+1)/(dim+1);
    return Fave


def aveCycleFidelity(kraus_V, kraus_M, U2):
    '''
    Takes as input:
        kraus_V: Kraus V for the 2-qubit error channel
        kraus_M: for single qubit noise 
        U2: 2-qubit channel (example is iSWAP) 
       
    Output:
        ave_cyc_fidelity: average cycle fidelity 
    '''
    dim=4;
    mu=0;
    for i in range(len(kraus_M[1,1,:])):
        mu = mu + (np.trace(kraus_M[:,:,i] @ kraus_M[:,:,i].conj().T)/3) - abs(np.trace(kraus_M[:,:,i]))**2/6;
    
    eta=0;
    for i in range(len(kraus_M[1,1,:])):
        eta = eta + abs(np.trace(kraus_M[:,:,i]))**2/3 - (np.trace(kraus_M[:,:,i] @ kraus_M[:,:,i].conj().T)/6);
    
    
    #l1=np.zeros([4,4],dtype=complex);
    #for i in range(len(kraus_V[1,1,:])):
    #    l1 = l1 + kraus_V[:,:,i] @ kraus_V[:,:,i].conj().T;
    #E1 = (mu**2/4)*np.trace(l1);
    E1 = (mu**2);

    vec_1q = np.eye(2);
    temp2 = 0;
    for i in range(len(kraus_V[1,1,:])):
        for k1 in range(2):
            for k2 in range(2):
                for l1 in range(2):
                    for l2 in range(2):
                        temp2 = temp2 + (np.kron(vec_1q[:,k1], vec_1q[:,k2]) @ (U2 @ kraus_V[:,:,i].conj().T) @ np.kron(vec_1q[:,l1], vec_1q[:,k2]))*(np.kron(vec_1q[:,l1], vec_1q[:,l2])@(kraus_V[:,:,i] @ U2.conj().T) @ np.kron(vec_1q[:,k1], vec_1q[:,l2]));
    E2 = (mu * eta/dim)*temp2;
    
    temp3 = 0;
    for i in range(len(kraus_V[1,1,:])):
        for k1 in range(2):
            for k2 in range(2):
                for l1 in range(2):
                    for l2 in range(2):
                        temp3 = temp3 + (np.kron(vec_1q[:,l1], vec_1q[:,l2])@(kraus_V[:,:,i] @ U2.conj().T) @ np.kron(vec_1q[:,l1], vec_1q[:,k2]))*(np.kron(vec_1q[:,k1], vec_1q[:,k2])@(U2 @ kraus_V[:,:,i].conj().T) @ np.kron(vec_1q[:,k1], vec_1q[:,l2]));
    E3 = (mu * eta/dim)*temp3;
    
    temp4=0;
    for i in range(len(kraus_V[1,1,:])):
        temp4 = temp4 + abs(np.trace(kraus_V[:,:,i] @ U2.conj().T))**2;
    E4 = (eta**2/4)*temp4;
    
    E = np.real(E1+E2+E3+E4);
    ave_cyc_fidelity = (E+1)/(dim+1);
    
    return ave_cyc_fidelity

def G1G2aveCycleFidelity(kraus_V, kraus_M, U2, G1, G2):
    '''
    Takes as input:
        kraus_V: Kraus V for the 2-qubit error channel
        kraus_M: for single qubit noise 
        U2: 2-qubit channel (example is iSWAP) 
        G1 & G2: single qubit unitaries 2x2 
       
    Output:
        G1G2_ave_cyc_fidelity: average cycle fidelity 
    '''
    
    dim=4;
    chi=0;
    for c in range(len(kraus_V[1,1,:])):
        for a in range(len(kraus_M[1,1,:])):
            for b in range(len(kraus_M[1,1,:])):
                chi = chi + abs(np.trace(np.kron(G1.conj().T @ kraus_M[:,:,a] @ G1, G2.conj().T @ kraus_M[:,:,b] @ G2) @ kraus_V[:,:,c] @ U2.conj().T))**2;
                
    chi = chi/dim;
    G1G2_ave_cyc_fidelity = (1+chi)/(1+dim);
    return G1G2_ave_cyc_fidelity
        
    
    
def logest(x, t, samples, freqExp, dPk):
    '''
    Takes as input:
        x: random varialbe
        t: time step value
        freqExp: input experimental frequency  
        dPk: input probability distribution  
       
    Output:
        A: value of the log-likelihood function 
    '''    

    A=0; 
    for i in range(samples):
        for k in range(4):
            A=A+freqExp[k,i, 0, t]*(dPk[k, i, t]/(x*dPk[k, i, t]+0.25));
        
    return A
    

    
def lmsEstimate(num_circ, num_measurements, depth, noise_param, noise_samples, method=True):

    '''
    Takes as input:
        num_circ: number of circuits 
        num_measurements: number of measurements
        depth: maximal depth of the circuit 
        noise_param: a dictionary containing all noise parameters 
        noise_samples: number of sample noises from uniform distribution 
        if there is control error
        
    Output:
        pD_star: list of pD_star parameter estimatesat each depth from 1 to maximal 
        ave_fid: average fidelity computed with Nielsen's formula 
        
    '''

    
    # Noise input parameters
    p1phase= noise_param['1q_phase']; # single qubit phase err. parameter 
    p1amp = noise_param['1q_amp']; # single qubit amp. err. parameter 
    p2dp = noise_param['2q_dep']; #depolarizing error probability 
    p2phase = noise_param['2q_phase']; # amplitude dampiing 
    p2amp = noise_param['2q_amp']; # amplitude dampiing 
    p2sw = noise_param['2q_iswap']; # width of the uniform distribution around pi/2 iSWAP error 


    # Internal Parameters 
    N=2; # number of qubits 
    eps = 10**(-16); # clipping P_U cutoff value 
    R_in = np.zeros([2**N,2**N],dtype=complex); R_in[0,0] = 1; # initial state all zero state for now 

    # Circuit string descriptions  
    CircStrings = np.zeros([num_circ, depth], dtype=int); # initialize array of zeros 
    for i in range(num_circ): 
        for j in range(depth): 
            CircStrings[i, j] = np.random.randint(len(gates[1,1,:])-1); # choose random number 0 to 5


    # Circuit Unitaries and Densitry Matrix Comptuation for noiseless circuit 
    CircU  = np.zeros([2**N,2**N, num_circ, depth+1], dtype=complex);
    CircR  = np.zeros([2**N,2**N, num_circ, depth+1], dtype=complex);
    for i in range(num_circ):
        CircU[:,:,i,0] = np.kron(I2, I2);
        CircR[:,:,i,0] = R_in;
        for j in range(1, depth+1):
            # compute unitary at each depth d single qubit pair + iSWAP
            CircU[:,:,i,j] = gates[:,:,len(gates[1,1,:])-1] @ gates[:,:, int(CircStrings[i, j-1])] @ CircU[:,:,i,j-1];
            # compute the density matrix at each depth d 
            CircR[:,:,i,j] = CircU[:,:,i,j] @ R_in @ CircU[:,:,i,j].conj().T;


    # Single-qubit total error (amplitude+phase) with parameter p1amp, p1phase 
    E0 = np.array([[np.sqrt(1-p1phase), 0 ],[0, np.sqrt(1-p1amp)*np.sqrt(1-p1phase)]]);
    E1 = np.array([[np.sqrt(p1phase), 0],[0, 0]]);
    E2 = np.array([[0, 0], [0, np.sqrt(p1phase)*np.sqrt(1-p1amp)]]);
    E3 = np.array([[0, np.sqrt(1-p1phase)*np.sqrt(p1amp)],[0, 0]]);
    E = np.zeros([2,2,4], dtype=complex); E[:,:,0] = E0; E[:,:,1] = E1; E[:,:,2] = E2; E[:,:,3] = E3;
    EL = np.zeros([4,4,4],dtype=complex); EL[:,:,0] = np.kron(E0,I2); EL[:,:,1] = np.kron(E1,I2); EL[:,:,2] = np.kron(E2,I2); EL[:,:,3] = np.kron(E3,I2);
    ER = np.zeros([4,4,4],dtype=complex); ER[:,:,0] = np.kron(I2,E0); ER[:,:,1] = np.kron(I2, E1); ER[:,:,2] = np.kron(I2,E2);ER[:,:,3] = np.kron(I2,E3);

    
    # Two-qubit incoherent error (amplitude+phase damping) with parameters p2amp, p2phase 
    K0 = np.array([[np.sqrt(1-p2phase), 0 ],[0, np.sqrt(1-p2amp)*np.sqrt(1-p2phase)]]);
    K1 =  np.array([[np.sqrt(p2phase), 0],[0, 0]]);
    K2 = np.array([[0, 0], [0, np.sqrt(p2phase)*np.sqrt(1-p2amp)]]);
    K3 = np.array([[0, np.sqrt(1-p2phase)*np.sqrt(p2amp)],[0, 0]]);
    K = np.zeros([2,2,4], dtype=complex); K[:,:,0] = K0; K[:,:,1] = K1; K[:,:,2] = K2; K[:,:,3] = K3;
    ELR = np.zeros([4,4,16],dtype=complex); 
    k=0;
    for i in range(4):
        for j in range(4):
            ELR[:, :, k] = np.kron(K[:,:,i], K[:,:,j]);
            k+=1;

            

    # Density matrix evolution under sigle-qubit and two-qubit errors 
    ErrR = np.zeros([4,4, num_circ, noise_samples, depth+1], dtype=complex); # initialize the denisty matrix to zeros

    for i in range(num_circ):
        for l in range(noise_samples):
            ErrR[:,:,i, l, 0] = R_in; # for each sample set it to be R_initial provided 
            for j in range(1, depth+1):
                tempR = np.zeros([4,4], dtype=complex);
                tempR = gates[:,:,CircStrings[i, j-1]] @ ErrR[:,:,i,l,j-1] @ gates[:,:,CircStrings[i, j-1]].conj().T;
                tempR = channel(tempR, ELR);
                tempR = iswap(np.random.uniform(-p2sw/2, p2sw/2)) @ tempR @ iswap(np.random.uniform(-p2sw/2, p2sw/2)).conj().T;
                ErrR[:,:,i, l, j] = dpchannel(tempR, p2dp); # apply two-qubit deloparizing error 


    # Cross-Entropy Computation 
    #(This part is not public and has been removed. 
    # The outputs are set to be empty.)
    
    freqExp = np.zeros([4, num_circ,noise_samples,depth+1]);
    dPk = np.zeros([4, num_circ, depth+1], dtype=complex);
    Pd = np.zeros([depth+1]);
    aveCycleFid = 1;
   
    return (freqExp, dPk, Pd, aveCycleFid)
   

meanMeasure = np.zeros ([10**4]);
stdMeasure = np.zeros([10**4]);
pEst = np.zeros([10**4]);
pErr = np.zeros([10**4]);
num_measurements = 1000;
depth=200;
num_circ =100;
noise_samples=1;
method=True;
noise_param = {
    '1q_phase': 0,
    '1q_amp': 0,
    '2q_dep': 0,
    '2q_phase':0.001,
    '2q_amp': 0.004,
    '2q_iswap': 0,
}


(freqExp, dPk, Pd, aveCycleFid) = lmsEstimate(num_circ, num_measurements, depth, noise_param, noise_samples, method)

    