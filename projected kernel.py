#adhoc dataset
adhoc_dimension = 2
X_train, y_train, X_test, y_test, adhoc_total = ad_hoc_data(
    training_size=20,
    test_size=5,
    n=adhoc_dimension,
    gap=0.3,
    plot_data=False,
    one_hot=False,
    include_sample_total=True,
)

feature1=X_train[:,0]
feature2=X_train[:,1]
feature3=X_test[:,0]
feature4=X_test[:,1]

#circuit
from qiskit.circuit import Parameter
rho1=[]
for i in range(len(feature1)):
      qc = QuantumCircuit(2)
      #l=2
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature1[i],[0])
      qc.rz(2*feature2[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[1])
      qc.cx(0,1)
      #
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature1[i],[0])
      qc.rz(2*feature2[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[1])
      qc.cx(0,1)
      #
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho1.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho1)
print(qc)

#circuit
from qiskit.circuit import Parameter
rho2=[]
for i in range(len(feature1)):
      qc = QuantumCircuit(2)
      #l=2
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature1[i],[0])
      qc.rz(2*feature2[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[1])
      qc.cx(0,1)
      #
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature1[i],[0])
      qc.rz(2*feature2[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[1])
      qc.cx(0,1)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[1])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho2.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho2)
print(qc)


temp = np.zeros([40,40]);
for i in range(40):
    for j in range(40):
        temp[i][j] = np.linalg.norm(rho1[i]-rho1[j], ord='fro')**2 +  np.linalg.norm(rho2[i]-rho2[j], ord='fro')**2;
gamma = 0.01;
Kernel = np.exp(-gamma * temp);
print(Kernel)


#circuit_test
from qiskit.circuit import Parameter
rho3=[]
for i in range(len(feature3)):
      qc = QuantumCircuit(2)
      #l=2
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature3[i],[0])
      qc.rz(2*feature4[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[1])
      qc.cx(0,1)
      #
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature3[i],[0])
      qc.rz(2*feature4[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[1])
      qc.cx(0,1)
      #
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[0])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho3.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho3)
print(qc)

#circuit_test
from qiskit.circuit import Parameter
rho4=[]
for i in range(len(feature3)):
      qc = QuantumCircuit(2)
      #l=2
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature3[i],[0])
      qc.rz(2*feature4[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[1])
      qc.cx(0,1)
      #
      qc.h(0)
      qc.h(1)
      qc.rz(2*feature3[i],[0])
      qc.rz(2*feature4[i],[1])
      qc.cx(0,1)
      qc.rz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[1])
      qc.cx(0,1)
      #
      t=time.time()
      qst_circuit = state_tomography_circuits(qc,[1])
      job = qiskit.execute(qst_circuit, Aer.get_backend('qasm_simulator'), shots=8000)
      tomo_circuit = StateTomographyFitter(job.result(), qst_circuit)
      rho_circuit = tomo_circuit.fit()
      rho4.append(rho_circuit)
print('Time taken:', time.time() - t)  
print(rho4)
print(qc)

temp_test = np.zeros([10,40]);
for i in range(10):
    for j in range(40):
        temp_test[i][j] = np.linalg.norm(rho1[j]-rho3[i], ord='fro')**2 +  np.linalg.norm(rho2[j]-rho4[i], ord='fro')**2;

Kernel_test = np.exp(-gamma * temp_test);
print(Kernel_test)


#Test the model
gram_test = Kernel_test
Y_predict_test=clf.predict(gram_test)

accuracy_score(Y_predict_test, y_test)
clf.score((gram_test),y_test)