import torch
import torch.nn as nn
from torchvision import models
from qiskit import QuantumCircuit
from qiskit_aer import Aer
import numpy as np

class QuantumLayer(nn.Module):
    def __init__(self, n_qubits=2):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, x):
        from qiskit import QuantumCircuit
        from qiskit_aer import Aer
        outputs = []
        for features in x:
            circuit = QuantumCircuit(self.n_qubits)
            for i in range(self.n_qubits):
                circuit.ry(float(features[i]), i)
            for i in range(self.n_qubits - 1):
                circuit.cx(i, i + 1)
            circuit.save_statevector()
            simulator = Aer.get_backend('aer_simulator_statevector')
            result = simulator.run(circuit).result()
            statevector = result.get_statevector()
            prob_0 = np.abs(statevector[0]) ** 2
            outputs.append([prob_0])
        return torch.tensor(outputs, dtype=torch.float32)

class QVTModel(nn.Module):
    def __init__(self, n_qubits=2):
        super().__init__()
        self.feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Identity()
        self.quantum = QuantumLayer(n_qubits=n_qubits)
        self.classifier = nn.Sequential(
            nn.Linear(1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        quantum_out = self.quantum(features[:, :2])
        out = self.classifier(quantum_out)
        return out