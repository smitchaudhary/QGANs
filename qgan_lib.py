import numpy as np
from components import *

lr_dis = 0.01
lr_gen = 0.01

class Generator:
    """
    A class to represent the generator.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the system.
    circ : Circuit
        Circuit of the generator.

    Methods
    -------
    calculate_gradient :
        Gives gradient of the circuit.
    update_params :
        Update parameters of the circuit.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.circ = Circuit(n_qubits)

    def calculate_gradient(self, dis):
        """
        Gives gradient of the circuit.

        Parameters
        ----------
        dis : Discriminator
            The Discriminator of the GAN.

        Returns
        -------
        grad : np.ndarray
            Gradient of the circuit.
        """
        weig_pauli = dis.weighted_paulis()

        init_state = initial_state(self.n_qubits)
        fake_state = np.matmul(self.circ.circ_matrix(), init_state)

        gradients = []

        ans = []

        for gate in self.circ.gates:
            if gate.id == 'CNOT':
                continue
            gradients.append(self.circ.grad_matrix(gate))

        for grad_i in gradients:
            fake_grad = np.matmul( grad_i, init_state )
            scal_grad = np.matmul( fake_grad.getH(), np.matmul( weig_pauli, fake_state ) ) + np.matmul( fake_state.getH(), np.matmul( weig_pauli, fake_grad ) )

            ans.append(scal_grad.item())

        ans = np.asarray(ans)
        grad = np.real(ans)
        return grad

    def update_params(self, dis):
        """
        Updates parameters of the circuit.

        Parameters
        ----------
        dis : Discriminator
            The Discriminator of the GAN.

        Returns
        -------
        None
        """
        gradients = self.calculate_gradient(dis)
        index = 0
        for gate in self.circ.gates:
            if gate.id == 'CNOT':
                continue
            gate.angle += lr_gen*gradients[index]
            index += 1

class Discriminator:
    """
    A class to represent the Discriminator.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the system.
    alpha : np.ndarray
        The weights of Paulis for real state

    Methods
    -------
    randomize_disc :
        Randomizes the parameters alphas
    weighted_paulis :
        Returns the full matrix for weighted Pauli strings.
    grad_pauli :
        Returns gradients with respect to one of the 4 paulis.
    grad_alpha :
        Gradient with respect to alphas.
    update_params :
        Update parameters of the Discriminator.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.alpha = np.zeros((n_qubits, 4))

    def randomize_disc(self):
        """
        Randomizes the parameters alphas

        Parameters
        ----------

        Returns
        -------
        None
        """
        for i in range(self.n_qubits):
            self.alpha[i] = -1 + 2*np.random.random(4)

    def weighted_paulis(self):
        """
        Returns the full matrix for weighted Pauli strings.

        Parameters
        ----------

        Returns
        -------
        ans : np.ndarray
            Full matrix for the weighted Pauli strings
        """
        ans = 1
        for i in range(self.n_qubits):
            mat = np.zeros_like(Y) # An array of zeros with size same as Y. Gave Y because data type complex
            for j in range(4):
                mat += self.alpha[i][j]*paulis[j]
            ans = np.kron(ans, mat)
        return ans

    def grad_pauli(self, pauli):
        """
        Returns gradients with respect to one of the 4 paulis.

        Parameters
        ----------
        pauli : np.ndarray
            One of the 4 Pauli gates.

        Returns
        -------
        ans : list
            Returns matrices with gradients with respect to parameters of the given Pauli.
        """
        params = self.alpha
        ans = []
        for i in range(self.n_qubits):
            ans_matrix = [1]
            for j in range(self.n_qubits):
                if i == j:
                    mat = pauli
                else:
                    mat = np.zeros_like(Y)
                    for k, gate in enumerate(paulis):
                        mat += params[j][k]*gate
                ans_matrix = np.kron(ans_matrix, mat)
            ans.append(ans_matrix)
        return ans

    def grad_alpha(self, gen, real_state):
        """
        Gradient with respect to alphas.

        Parameters
        ----------
        gen : Generator
            The Generator of the GAN.
        real_state : np.ndarray
            The real state.

        Returns
        -------
        ans : np.ndarray
            Gradient with respect to alphas.
        """
        weig_pauli = self.weighted_paulis()
        init_state = initial_state(self.n_qubits)
        fake_state = np.matmul(gen.circ.circ_matrix(), init_state)
        state = real_state - fake_state
        ans = np.zeros_like(self.alpha, dtype = complex)
        for index, pauli in enumerate(paulis):
            grads = self.grad_pauli(pauli)
            grad_list = []

            for grad_i in grads:
                rl = np.matmul(real_state.getH(), np.matmul(grad_i, real_state) ).item()
                fk = np.matmul(fake_state.getH(), np.matmul(grad_i, fake_state) ).item()
                grad_list.append(np.matmul(state.getH(), np.matmul(grad_i, state) ).item() )

            ans[:, index] = np.asarray(grad_list)

        return np.around(np.real(ans),6)

    def update_params(self, gen, real_state):
        """
        Updates parameters of the circuit.

        Parameters
        ----------
        gen : Generator
            The Generator of the GAN.
        real_state : np.ndarray
            The real state.

        Returns
        -------
        None
        """
        self.alpha += lr_dis*self.grad_alpha(gen, real_state)
        self.alpha = self.alpha/(np.max(np.abs(self.alpha)))


def fidelity(gen, real_state):
    """
    Computes fidelity between state produced by the generator and the real state.

    Parameters
    ----------
    gen : Generator
        The Generator of the GAN.
    real_state : np.ndarray
        The real state.

    Returns
    -------
    fid : float
        The fidelity between state produced by the generator and the real state.
    """
    init_state = initial_state(gen.n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)
    return np.abs(np.matmul(real_state.getH(), fake_state).item()  )**2

def compute_cost(gen, dis, real_state):
    """
    Computes cost.

    Parameters
    ----------
    gen : Generator
        The Generator of the GAN.
    dis : Discriminator
        The Discriminator of the GAN.
    real_state : np.ndarray
        The real state.

    Returns
    -------
     : float
        The cost
    """
    alpha = dis.alpha
    n_qubits = gen.n_qubits
    weig_pauli = dis.weighted_paulis()

    init_state = initial_state(n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)

    real_pauli_expec = np.matmul( real_state.getH(), np.matmul(weig_pauli, real_state) ).item()
    fake_pauli_expec = np.matmul( fake_state.getH(), np.matmul(weig_pauli, fake_state) ).item()

    return np.real(real_pauli_expec - fake_pauli_expec)

def gen_cost(gen, dis):
    """
    Computes cost.

    Parameters
    ----------
    gen : Generator
        The Generator of the GAN.
    dis : Discriminator
        The Discriminator of the GAN.
    real_state : np.ndarray
        The real state.

    Returns
    -------
     : float
        The cost
    """
    alpha = dis.alpha
    n_qubits = gen.n_qubits
    weig_pauli = dis.weighted_paulis()

    init_state = initial_state(n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)

    fake_pauli_expec = np.matmul( fake_state.getH(), np.matmul(weig_pauli, fake_state) ).item()

    return np.real(fake_pauli_expec)
