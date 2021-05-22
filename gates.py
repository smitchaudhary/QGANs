import numpy as np
import scipy.linalg as linalg

I = np.eye(2)
X = np.matrix([ [0, 1],
                [1, 0] ])
Y = np.matrix([ [0, -1j],
                [1j, 0] ])
Z = np.matrix([ [1, 0],
                [0, -1] ])

zero = np.matrix([ [1, 0],
                    [0, 0]] )
one = np.matrix([[ 0, 0 ],
                 [ 0, 1 ]] )


H = (1/np.sqrt(2))*np.matrix([ [1, 1],
                                [1, -1] ])

CNOT = np.matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])

def I_n(n):
    return np.eye(2**n)

def single_qubit_rotation(n_qubits, target_qubit, angle, axis):
    if axis == 'X':
        rot = linalg.expm(-0.5j*angle*X)
    elif axis == 'Y':
        rot = linalg.expm(-0.5j*angle*Y)
    else:
        rot = linalg.expm(-0.5j*angle*Z)
    ans = 1
    for i in range(n_qubits):
        if i == target_qubit:
            ans = np.kron(ans, rot)
        else:
            ans = np.kron(ans, I)
    return ans

def full_CNOT(n_qubits, control, target):
    temp1 = np.kron( I_n(control), np.kron( zero, I_n(n_qubits - control - 1) ) )
    temp2 = np.kron(I_n(control), np.kron(one, np.kron(I_n(target - control - 1), np.kron(X, I_n(n_qubits - target - 1)))))
    return temp1 + temp2
