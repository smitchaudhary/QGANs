import numpy as np
from gates import *
lr = 0.005


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
        real_dis, fake_dis = dis.real_and_fake_part()
        init_state = initial_state(self.n_qubits)
        fake_state = np.matmul(self.circ.circ_matrix(), init_state)

        gradients = []

        ans = []

        for gate_index in range(len(self.circ.gates)):
            gradients.append(self.circ.grad_matrix(gate_index))

        for grad_i in gradients:
            fake_grad = np.matmul( grad_i, init_state )
            scal_grad = np.matmul( fake_grad.getH(), np.matmul( fake_dis, fake_state ) ) + np.matmul( fake_state.getH(), np.matmul( fake_dis, fake_grad ) )

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
        for index, gate in enumerate(self.circ.gates):
            gate.angle += lr*gradients[index]
            #print(f' Gradient is {gradients[index]}')

class Discriminator:
    """
    A class to represent the Discriminator.

    Attributes
    ----------
    n_qubits : int
        Number of qubits in the system.
    alpha : np.ndarray
        The weights of Paulis for real state
    beta : np.ndarray
        The weights of Paulis for fake state

    Methods
    -------
    randomize_disc :
        Randomizes alphas and betas
    real_and_fake_part :
        Returns real and fake weighted Pauli strings.
    grad_real_fake :
        Returns gradients of real and fake weighted paulis.
    grad_alpha_beta :
        Gradient with respect to alphas and betas.
    update_params :
        Update parameters of the Discriminator.
    """
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.alpha = np.zeros((n_qubits, 4))
        self.beta = np.zeros((n_qubits, 4))

    def randomize_disc(self):
        """
        Randomizes alphas and betas

        Parameters
        ----------

        Returns
        -------
        None
        """
        for i in range(self.n_qubits):
            self.alpha[i] = -1 + 2*np.random.random(4)
            self.beta[i] = -1 + 2*np.random.random(4)
        print(self.alpha)
        print(self.beta)

    def real_and_fake_part(self):
        """
        Returns real and fake weighted Pauli strings.

        Parameters
        ----------

        Returns
        -------
        real_dis : np.ndarray
            Weighted real Pauli string
        fake_dis : np.ndarray
            Weighted fake Pauli string
        """
        real_dis = 1
        fake_dis = 1
        for i in range(self.n_qubits):
            real = np.zeros_like(Y) # An array of zeros with size same as Y. Gave Y because data type complex
            fake = np.zeros_like(Y)
            for j in range(4):
                real += self.alpha[i][j]*paulis[j]
                fake += self.beta[i][j]*paulis[j]
            real_dis = np.kron(real_dis, real)
            fake_dis = np.kron(fake_dis, fake)
        return real_dis, fake_dis

    def grad_real_fake(self, pauli, real_bool = True):
        """
        Returns gradients of real and fake weighted paulis.

        Parameters
        ----------
        pauli : np.ndarray
            One of the 4 Pauli gates.
        real_bool = bool
            True if we want gradient wrt real, False for fake. Defaults to True.

        Returns
        -------
        ans : np.ndarray
            Gradient with respect to real or fake
        """
        if real_bool:
            params = self.alpha
        else:
            params = self.beta
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

    def grad_alpha_beta(self, gen, real_state, real_bool = True):
        """
        Gradient with respect to alphas or betas.

        Parameters
        ----------
        gen : Generator
            The Generator of the GAN.
        real_state : np.ndarray
            The real state.
        real_bool = bool
            True if we want gradient wrt alphas, False for betas. Defaults to True.

        Returns
        -------
        ans : np.ndarray
            Gradient with respect to alphas or betas
        """
        real_dis, fake_dis = self.real_and_fake_part()
        if real_bool:
            state = real_state
            scal = 1
        else:
            init_state = initial_state(self.n_qubits)
            state = np.matmul(gen.circ.circ_matrix(), init_state)
            scal = -1
        ans = np.zeros_like(self.alpha, dtype = complex)
        for index, pauli in enumerate(paulis):
            grads = self.grad_real_fake(pauli, real_bool)
            grad_list = []

            for grad_i in grads:
                grad_list.append(np.matmul(state.getH(), np.matmul(grad_i, state) ).item() )

            ans[:, index] = np.asarray(grad_list)

        return np.around(scal*np.real(ans),6)

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
        self.alpha -= lr*self.grad_alpha_beta(gen, real_state, real_bool = True)
        self.beta +- lr*self.grad_alpha_beta(gen, real_state, real_bool = False)
        self.alpha = self.alpha/(np.max(np.abs(self.alpha)))
        self.beta = self.beta/(np.max(np.abs(self.beta)))


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
    beta = dis.beta
    n_qubits = gen.n_qubits
    real_dis, fake_dis = dis.real_and_fake_part()

    init_state = initial_state(n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)

    real_pauli_expec = np.matmul( real_state.getH(), np.matmul(real_dis, real_state) ).item()
    fake_pauli_expec = np.matmul( fake_state.getH(), np.matmul(fake_dis, fake_state) ).item()

    return np.real(real_pauli_expec - fake_pauli_expec)
