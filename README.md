# The power of one clean qubit in supervised machine learning.

This repo contains the codes used to generate the results of [this paper](https://arxiv.org/abs/2210.09275).
We utilize DQC1 as a kernel for binary classification. We implement our experiment on the IBM quantum hardware. Furthermore, we address the quantum discord and
coherence consumption in our DQC1 algorithm, both theoretically and experimentally.

## Required Modules
It is required to have the following modules installed before running the codes
- qiskit
- qiskit-quantum-machine-learning

## Codes
There are 5 files used for each part of the experiment. The Python libraries used could be found in [libraries.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/libraries.py).
Circuit implementation is done in [circuit.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/circuit.py).
The tomography procedure to reconstruct the output state is coded in [tomography.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/tomography.py).
Quantum kernel codes are provided in Kernel [preparation.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/Kernel%20preparation.py).
The plots are produced by [plots.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/plots.py).
After that, changing the parameter \alpha has been shown in [alpha.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/alpha.py), you can run the code for different values of \alpha and get accuracy vs.\alpha plots, for different datasets like ``ad-hoc'', ``make-moon'' and ``make-circle''.

Overall, to get the outputs, one could simply run [code.py](https://github.com/mahsakarimii/The-power-of-one-clean-qubit-in-supervised-machine-learning./blob/main/code.py).

## Outputs
Running the code, we get the following:

- Classification

![download](https://user-images.githubusercontent.com/67652297/208560700-3db77d75-d8cc-4b62-aa7f-782e8e7097c6.png)


- Quantum Kernel

![download](https://user-images.githubusercontent.com/67652297/208560881-e0edc2c6-d43c-490a-84e7-9b88523d7546.png)


- Coherence

![download](https://user-images.githubusercontent.com/67652297/208560931-cfcbc7b1-f2c6-42d1-8c87-bf836f22db78.png)


- Discord

![download](https://user-images.githubusercontent.com/67652297/208561011-01ac8be5-0d76-4ed2-a15f-62f7286b52a4.png)

You can also do the same on IBM real hardware using your IBM account. Create one on [IBM's webpage](https://quantum-computing.ibm.com/) if you do not have any :)


