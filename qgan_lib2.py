import numpy as np
from gates import *

class Generator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.circ = Circuit(n_qubits)

    def calculate_gradient(self, dis):
        real_dis, fake_dis = real_and_fake_part(dis)
        init_state = initial_state(self.n_qubits)
        fake_state = np.matmul(gen.circ.circ_matrix(), init_state)

        gradients = []

        ans = []

        for gate_index in range(len(self.circ.gates)):
            gradients.append(self.circ.grad_matrix())

        for grad_i in gradients:
            fake_grad = np.matmul( grad_i, init_state )
            scal_grad = np.matmul( fake_grad.getH(), np.matmul( fake_dis, fake_states ) ) + np.matmul( fake_state.getH(), np.matmul( fake_dis, fake_grad ) )

            ans.append(np.asscalar(scal_grad))

        ans = np.asarray(ans)
        grad = np.real(ans)
        return grad

    def update_params(self, dis):
        gradients = self.calculate_gradient(dis)
        for index, gate in self.circ.gates:
            gate.angle -= lr*gradients[index]

class Discriminator:
    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.alpha = np.zeros((n_qubits, 4))
        self.beta = np.zeros((n_qubits, 4))

    def randomize_disc(self):
        for i in range(self.n_qubits):
            self.alpha[i] = -1 + 2*np.random.random(4)
            self.beta[i] = -1 + 2*np.random.random(4)

    def real_and_fake_part(self):
        paulis = [I, X, Y, Z]
        real_dis = 1
        fake_dis = 1
        for i in range(n_qubits):
            real = np.zeros_like(Y) # An array of zeros with size same as Y. Gave Y because data type complex
            fake = np.zeros_like(Y)
            for j in range(4):
                real += alpha[i][j]*paulis[j]
                fake += beta[i][j]*paulis[j]
            real_dis = np.kron(real_dis, real)
            fake_dis = np.kron(fake_dis, fake)
        return real_dis, fake_dis

    def grad_real_fake(self, pauli, real_bool = True):
        paulis = [I, X, Y, Z]
        if real_bool:
            params = self.alpha
        else:
            params = self.beta
        ans = []
        for i in range(self.n_qubits):
            ans_matrix = 1
            for j in range(self.n_qubits):
                if i == j:
                    mat = pauli
                else:
                    mat = np.zeros_like(Y)
                    for gate in paulis:
                        mat += params[j][k]*gate
                ans_matrix = np.kron(ans_matrix, mat)
            ans.append(ans_matrix)
        return ans

    def grad_alpha_beta(self, gen, real_state, real_bool = True):
        real_dis, fake_dis = self.real_and_fake_part()
        if real_bool:
            state = real_state
            scal = 1
        else:
            init_state = initial_state(n_qubits)
            fake_state = np.matmul(gen.circ.circ_matrix(), init_state)
            scal = -1
        paulis = [I, X, Y, Z]
        ans = np.zeros_like(self.alpha, dtype = complex)
        for index, pauli in enumerate(paulis):
            grads = self.grad_real_fake(type, real_bool)
            grad_list = []

            for grad_i in grads:
                grad_list.append(np.matmul(state.getH(), np.matmul(grad_i, state) ) )

            ans[:, index] = np.asarray(grad_list)

        return scal*np.real(ans)

    def update_params(self, gen, real_state):
        self.alpha += lr*self.grad_alpha_beta(gen, real_state, real_bool = True)
        self.beta += lr*self.grad_alpha_beta(gen, real_state, real_bool = False)


def fidelity(gen, real_state):
    init_state = initial_state(gen.n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)
    return np.abs(np.asscalar(np.matmul(real_state.getH(), fake_state)))**2

def compute_cost(gen, dis, real_state):
    alpha = dis.alpha
    beta = dis.beta
    n_qubits = gen.n_qubits
    paulis = [I, X, Y, Z]
    real_dis, fake_dis = real_and_fake_part(dis)

    init_state = initial_state(n_qubits)
    fake_state = np.matmul(gen.circ.circ_matrix(), init_state)

    real_pauli_expec = np.asscalar(np.matmul( real_state.getH(), np.matmul(real_dis, real_state) ) )
    fake_pauli_expec = np.asscalar(np.matmul( fake_state.getH(), np.matmul(fake_dis, fake_state) ) )

    return real_pauli_expec - fake_pauli_expec
