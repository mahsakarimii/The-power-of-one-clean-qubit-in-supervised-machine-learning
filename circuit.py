#adhoc dataset
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


feature1=train_features[:,0]
feature2=train_features[:,1]

feature3=test_features[:,0]
feature4=test_features[:,1]

qc = QuantumCircuit(5)
#circuit preparation
qc.h(0)
qc.h(1)
qc.cx(1,3)
qc.h(2)
qc.cx(2,4)

#training_circuit

#Controlled-$U$
qc.ch(0,1)
qc.ch(0,2)
qc.crz(2*feature1[i],[0],[1])
qc.crz(2*feature2[i],[0],[2])
qc.ccx(0,1,2)
qc.crz(2*(np.pi-feature1[i])*(np.pi-feature2[i]),[0],[2])
qc.ccx(0,1,2)
#Controlled-$U\dagger$
qc.ccx(0,1,2)
qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
qc.ccx(0,1,2)
qc.crz(-2*feature2[j],[0],[2])
qc.crz(-2*feature1[j],[0],[1])
qc.ch(0,2)
qc.ch(0,1)


#testing_circuit

#Controlled-$U$
qc.ch(0,1)
qc.ch(0,2)
qc.crz(2*feature3[i],[0],[1])
qc.crz(2*feature4[i],[0],[2])
qc.ccx(0,1,2)
qc.crz(2*(np.pi-feature3[i])*(np.pi-feature4[i]),[0],[2])
qc.ccx(0,1,2)
#Controlled-$U\dagger$
qc.ccx(0,1,2)
qc.crz(-2*(np.pi-feature1[j])*(np.pi-feature2[j]),[0],[2])
qc.ccx(0,1,2)
qc.crz(-2*feature2[j],[0],[2])
c.crz(-2*feature1[j],[0],[1])
qc.ch(0,2)
qc.ch(0,1)