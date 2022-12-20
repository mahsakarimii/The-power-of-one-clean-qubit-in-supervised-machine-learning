import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.svm import SVC

from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from qiskit import BasicAer
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit import *


# Tomography functions
from qiskit.ignis.verification.tomography import state_tomography_circuits, StateTomographyFitter
import qiskit.ignis.mitigation.measurement as mc

# example dataset
import numpy as np
from qiskit import QuantumCircuit, BasicAer, transpile
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.datasets import ad_hoc_data

adhoc_dimension = 2
train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)
#Then you can repeat everything with other datasets like make_moon and make_circle in scikit-learn.

feature1=train_features[:,0]
feature2=train_features[:,1]
feature3=test_features[:,0]
feature4=test_features[:,1]


from qiskit.circuit import Parameter
rho=[]
for i in range(len(train_features)):
  for j in range(len(train_features)):
      qc = QuantumCircuit(6)
      #There is a formula between \alpha and \theta, by tunning \theta(\alpha), you can get desired \alpha.
      #For instance, here \alpha=0 which is corresponding to random guessing in ML.
      qc.ry(2*np.arccos(np.sqrt(1)),0)
      qc.cx(0,5)
      qc.h(0)
      qc.h(1)
      qc.cx(1,3)
      qc.h(2)
      qc.cx(2,4)
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[i])*(np.pi-feature2[i]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[i],[0],[2])
      qc.crz(-2*feature1[i],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature1[i])*(np.pi-feature2[i]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature2[i],[0],[2])
      qc.crz(-2*feature1[i],[0],[1])

      qc.crz(2*feature1[j],[0],[1])
      qc.crz(2*feature2[j],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature1[j],[0],[1])
      qc.crz(2*feature2[j],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho.append(rho_circuit)
print('Time taken:', time.time() - t)
print(qc)

ph=np.zeros(len(rho),dtype=np.complex_)
for i in range(len(rho)):
  ph[i]=rho[i][0][1]

pp=[]
for i in ph:
  pp.append(2*(np.abs(i)))

k = np.reshape(pp,(40,40))
d=[]
for i in range(40):
  d.append(k[i][i])

from qiskit.circuit import Parameter
rho_test=[]
for i in range(len(test_features)):
  for j in range(len(train_features)):
      qc = QuantumCircuit(6)
      qc.ry(2*np.arccos(np.sqrt(1)),0)
      qc.cx(0,5)
      qc.h(0)
      qc.h(1)
      qc.cx(1,3)
      qc.h(2)
      qc.cx(2,4)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature3[i])*(np.pi-feature4[i]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature4[i],[0],[2])
      qc.crz(-2*feature3[i],[0],[1])
      qc.ch(0,2)
      qc.ch(0,1)
      #
      qc.ccx(0,1,2)
      qc.crz(-2*(np.pi-feature3[i])*(np.pi-feature4[i]),[0],[2])
      qc.ccx(0,1,2)
      qc.crz(-2*feature4[i],[0],[2])
      qc.crz(-2*feature3[i],[0],[1])

      qc.crz(2*feature1[j],[0],[1])
      qc.crz(2*feature2[j],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      #
      qc.ch(0,1)
      qc.ch(0,2)
      qc.crz(2*feature1[j],[0],[1])
      qc.crz(2*feature2[j],[0],[2])
      qc.ccx(0,1,2)
      qc.crz(2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
      qc.ccx(0,1,2)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho_test.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(qc)

pht=np.zeros(len(rho_test),dtype=np.complex_)
for i in range(len(rho_test)):
  pht[i]=rho_test[i][0][1]

ppt=[]
for j in pht:
  ppt.append(2*(np.abs(j)))

kt = np.reshape(ppt,(10,40))

clf = svm.SVC(kernel='precomputed')
gram_train = k
Y_train=clf.fit(gram_train, train_labels)

gram_test = kt
#clf.predict(gram_test)
Y_predict=clf.predict(gram_test)

accuracy_score(Y_predict, test_labels)

clf.score(kt,test_labels)

# Now, you can repeat the process for different values of \alpha 
# (it has been shown in the code that how you can change it)
# and get the accuracy vs. \alpha plot.


