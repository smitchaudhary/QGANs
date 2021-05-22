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

def single_qubit_rotation(n_qubits, target_qubit, axis, angle = np.pi):
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

class Gate:
    def __init__(self, id, control = None, target = None, angle = np.pi):
        self.id = id
        self.control = control
        self.target = target
        self.angle = angle

    def full_matrix(self, n_qubits):
        if self.id == 'X' or self.id == 'Y':
            return 1j*single_qubit_rotation(n_qubits, self.target, self.id)
        if self.id =='CNOT':
            return full_CNOT(n_qubits, self.control, self.target)

class Circuit:
    def __init__(self, n_qubits):
        self.n_qubits =  n_qubits
        self.gates = []

    def circ_matrix(self):
        ans = I_n(n_qubits)
        for gate in self.gates:
            ans = np.matul( gate.full_matrix(self.n_qubits) )

        return np.asmatrix(ans)

    def append_gate(self, gate):
        self.gates.append(gate)

print(single_qubit_rotation(1, 0, 'X'))
