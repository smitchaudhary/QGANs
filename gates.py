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

CNOT = np.matrix([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0]])

def I_n(n):
    if n < 0:
        return np.array([[1]])
    dim = int(2**n)
    return np.eye(dim)

def single_qubit_rotation(n_qubits, target_qubit, axis, angle = np.pi, grad = False):
    """
    Gives the full 2**n x 2**n matrix for the single qubit rotation gate.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    target_qubit : int
        Index of the qubit on which the rotation is applied.
    axis : str
        Axis along which rotation occurs. Here, 'X', 'Y', 'Z' allowed.
    angle = float
        Angle by which the rotation occurs. Defaults to pi.
    grad : bool
        If true, full matrix of the gradient with respect to parameter.
        If false, full matrix.
        Defaults to False.

    Returns
    -------
    ans : np.ndarray
        The full 2**n x 2**n matrix for the single qubit rotation gate.
    """
    if grad == False:
        if axis == 'X':
            #print(f' Angle is {angle}')
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
    """
    Gives the full 2**n x 2**n matrix for the CNOT gate.

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.
    control : int
        Index of the control qubit.
    target_qubit : int
        Index of the target qubit.
    grad : bool
        If true, full matrix of the gradient with respect to parameter.
        If false, full matrix.
        Defaults to False.

    Returns
    -------
    temp1 + temp2 : np.ndarray
        The full 2**n x 2**n matrix for the CNOT gate.
    """
    if n_qubits == 2:
        return CNOT
    a, b = min(control, target), max(control, target)
    if a == control:
        temp1 = np.kron( I_n(control), np.kron( zero, I_n(n_qubits - control - 1) ) )
        temp2 = np.kron(I_n(control), np.kron(one, np.kron(I_n(target - control - 1), np.kron(X, I_n(n_qubits - target - 1)))))
    else:
        temp1 = np.kron( I_n(control), np.kron( zero, I_n(n_qubits - control - 1) ) )
        temp2 = np.kron(I_n(target), np.kron(X, np.kron(I_n(control - target - 1), np.kron(one, I_n(n_qubits - control - 1)))))
    return temp1 + temp2

class Gate:
    """
    A class to represent a gate.

    Attributes
    ----------
    id : str
        Which gate it is. Rotation about X, rotation about Z and CNOT allowed.
    control : int
        Indiex of the control qubit. Defaults to none in case of single qubit gates.
    target : int
        Index of the target qubit. Index of the qubit in case of single qubit rotations. Index of the target in case of CNOT.
    angle : float
        Angle by which the rotation occurs. Defaults to pi. Unused in case of CNOT gate.

    Methods
    -------
    full_matrix :
        Gives the full 2**n x 2**n matrix for the gate.
    """
    def __init__(self, id, control = None, target = None, angle = np.pi):
        self.id = id
        self.control = control
        self.target = target
        self.angle = angle

    def full_matrix(self, n_qubits, grad = False):
        """
        Gives the full 2**n x 2**n matrix for the gate.

        Parameters
        ----------
        n_qubits : int
            Number of qubits in the system.
        grad : bool
            If true, full matrix of the gradient with respect to parameter.
            If false, full matrix.
            Defaults to False.

        Returns
        -------
            : np.ndarray
            Gives the full 2**n x 2**n matrix for the gate.
        """
        if self.id == 'X' or self.id == 'Z':
            return 1j*single_qubit_rotation(n_qubits, self.target, self.id, angle = self.angle, grad = grad)
        if self.id =='CNOT':
            return full_CNOT(n_qubits, self.control, self.target, grad = grad)

class Circuit:
    """
    A class to represent a circuit.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the system.
    gates : list
        List of all gates (objects of class Gate) in the circuit.

    Methods
    -------
    circ_matrix :
        Gives the full 2**n x 2**n matrix for the circuit.
    grad_matrix :
        Gives the full 2**n x 2**n matrix for the gradient of the circuit with respect to parameter of a parametric gate.
    append_gate :
        Add gates to the circuit.
    randomize_angles :
        Randomize parametrs of the parametric gates.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.gates = []

    def circ_matrix(self):
        """
        Gives the full 2**n x 2**n matrix for the circuit.

        Parameters
        ----------

        Returns
        -------
        ans : np.matrix
            Gives the full 2**n x 2**n matrix for the circuit.
        """
        ans = I_n(self.n_qubits)
        for gate in self.gates:
            ans = np.matmul(gate.full_matrix(self.n_qubits), ans )

        return np.asmatrix(ans)

    def grad_matrix(self, gate_index):
        """
        Gives the full 2**n x 2**n matrix for the gradient of the circuit.

        Parameters
        ----------
        gate_index : int
            Index of the gate with respect ot whose paramter the gradient is taken.

        Returns
        -------
        ans : np.matrix
            Gives the full 2**n x 2**n matrix for the circuit.
        """
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
        """
        Adds gate to the circuit.

        Parameters
        ----------
        gate : Gate
            Gate that is to be added to the circuit.

        Returns
        -------
        None
        """
        self.gates.append(gate)

    def randomize_angles(self):
        """
        Randomize the parameters of the parametric gates.

        Parameters
        ----------

        Returns
        -------
        None
        """
        for gate in self.gates:
            gate.angle = 2*np.pi*np.random.random()
            print(gate.angle)
        print('Randomization done')

def initial_state(n_qubits):
    """
    Gives the zero state

    Parameters
    ----------
    n_qubits : int
        Number of qubits in the system.

    Returns
    -------
    state : np.array
        The zero state in vector form.
    """
    state = np.zeros(2**n_qubits)
    state[0] = 1
    state = np.asmatrix(state).T
    return state
#print(1j*single_qubit_rotation(1, 0, 'X', grad = True))
