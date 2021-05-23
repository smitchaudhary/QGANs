import numpy as np
import scipy.linalg as linalg

I = np.eye(2)
X = np.matrix([ [0, 1],
                [1, 0] ])
Y = np.matrix([ [0, -1j],
                [1j, 0] ])
Z = np.matrix([ [1, 0],
                [0, -1] ])

paulis = [I, X, Y, Z]

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
    dim = int(2**n)
    return np.eye(dim)

def single_qubit_rotation(n_qubits, target_qubit, axis, angle = np.pi, grad = False):
    if grad == False:
        if axis == 'X':
            rot = linalg.expm(-0.5j*angle*X)
        elif axis == 'Y':
            rot = linalg.expm(-0.5j*angle*Y)
        else:
            rot = linalg.expm(-0.5j*angle*Z)
    else:
        if axis == 'X':
            rot = -0.5j*X*linalg.expm(-0.5j*angle*X)
        elif axis == 'Y':
            rot = -0.5j*Y*linalg.expm(-0.5j*angle*Y)
        else:
            rot = -0.5j*Z*linalg.expm(-0.5j*angle*Z)
    ans = 1
    for i in range(n_qubits):
        if i == target_qubit:
            ans = np.kron(ans, rot)
        else:
            ans = np.kron(ans, I)
    return ans

def full_CNOT(n_qubits, control, target, grad = False):
    a, b = min(control, target), max(control, target)
    if a == control:
        temp1 = np.kron( I_n(control), np.kron( zero, I_n(n_qubits - control - 1) ) )
        temp2 = np.kron(I_n(control), np.kron(one, np.kron(I_n(target - control - 1), np.kron(X, I_n(n_qubits - target - 1)))))
    else:
        temp1 = np.kron( I_n(control), np.kron( zero, I_n(n_qubits - control - 1) ) )
        temp2 = np.kron(I_n(target), np.kron(one, np.kron(I_n(target - control - 1), np.kron(X, I_n(n_qubits - control - 1)))))
    return temp1 + temp2

class Gate:
    def __init__(self, id, control = None, target = None, angle = np.pi):
        self.id = id
        self.control = control
        self.target = target
        self.angle = angle

    def full_matrix(self, n_qubits, grad = False):
        if self.id == 'X' or self.id == 'Z':
            return 1j*single_qubit_rotation(n_qubits, self.target, self.id, angle = self.angle, grad = grad)
        if self.id =='CNOT':
            return full_CNOT(n_qubits, self.control, self.target, grad = grad)

class Circuit:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    def circ_matrix(self):
        ans = I_n(self.n_qubits)
        for gate in self.gates:
            ans = np.matmul(gate.full_matrix(self.n_qubits), ans )

        return np.asmatrix(ans)

    def grad_matrix(self, gate_index):
        ans = I_n(self.n_qubits)
        for index, gate in enumerate(self.gates):
            if index == gate_index:
                mat = gate.full_matrix(self.n_qubits, grad = True)
                ans = np.matmul(mat, ans)
            else:
                mat = gate.full_matrix(self.n_qubits, grad = False)
                ans = np.matmul(mat, ans)
        return ans

    def append_gate(self, gate):
        self.gates.append(gate)

def initial_state(n_qubits):
    state = np.zeros(2**n_qubits)
    state[0] = 1
    state = np.asmatrix(state).T
    return state
#print(1j*single_qubit_rotation(1, 0, 'X', grad = True))
