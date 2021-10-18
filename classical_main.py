import cirq
import numpy as np
from classical_lib import *

print(f'Setting up hyper parameters')
epochs = 1000
n_qubits = 1
gen_dis_ratio = 1
depth = 2
n_params = 2 * depth * n_qubits
shots = 0
lr_dis = 0.01

theta0 = np.random.random(n_params)*2*np.pi
qcbm = variational_circuit(n_qubits, depth, theta0)

training_data_length = 1024
train_data = torch.zeros((training_data_length, 2))

train_data[:,0] = torch.from_numpy(np.random.randint(0,2**n_qubits, size=training_data_length) )

real_probs = real_probabs(n_qubits)

for index, key in enumerate(train_data[:,0]):
    key = int(key)
    train_data[index,1] = real_probs[key]

train_labels = torch.zeros(training_data_length)
train_set = [(train_data[i], train_labels[i]) for i in range(training_data_length)]

train_loader = torch.utils.data.DataLoader(train_set, shuffle = True)

loss_function = nn.BCELoss()

dis = Discriminator()

gen = Generator(qcbm, theta0, n_qubits)

dis_optimizer = torch.optim.Adam(dis.parameters(), lr = lr_dis)

for i in range(epochs):
    for n, (real_samples, useless_label) in enumerate(train_loader):
        real_samples_labels = torch.ones((1,1))
        latent_space_samples = np.random.randint(0,2**n_qubits)
        generated_samples = torch.from_numpy(np.array(gen.forward(latent_space_samples)))
        generated_samples_labels = torch.zeros((1, 1))
        print(real_samples)
        print(generated_samples)
        all_samples = torch.cat((real_samples, generated_samples))
        all_samples_labels = torch.cat((real_samples_labels, generated_samples_labels))
