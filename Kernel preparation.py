#just picking the element \rho_{11} of all the density matrices.
ph=np.zeros(len(rho),dtype=np.complex_)
for i in range(len(rho)):
  ph[i]=rho[i][0][1]

#element \rho_{11}s are complex numbers with abs command we calculate their absolute value. 
# Here, coefficient two reffers to the 1/2 coefficient outside of the density matrix.
pp=[]
for i in ph:
  pp.append(2*(np.abs(i)))

#Let's put them into a 40*40 matrix (Gram matrix)
k = np.reshape(pp,(40,40))
#Let's print the diagonal elements to see wether they are equal two one or not.
d=[]
for i in range(40):
  d.append(k[i][i])

#just picking the element \rho_{11} of all the density matrices.
pht=np.zeros(len(rho_test),dtype=np.complex_)
for i in range(len(rho_test)):
  pht[i]=rho_test[i][0][1]


#element \rho_{11}s are complex numbers with abs command we calculate their absolute value. 
# Here, coefficient two reffers to the 1/2 coefficient outside of the density matrix.
ppt=[]
for j in pht:
  ppt.append(2*(np.abs(j)))


#Let's put them into a 10*40 matrix (Gram matrix for testing datasets)
kt = np.reshape(ppt,(10,40))

#Train the model
clf = svm.SVC(kernel='precomputed')
gram_train = k
Y_train=clf.fit(gram_train, train_labels)

#Test the model
gram_test = kt
Y_predict=clf.predict(gram_test)

accuracy_score(Y_predict, test_labels)



#Some mathematical formula to calculate coherence
x1=np.abs(1-k)/2


#mathematical formula to calculate discord
dis=1/16*(1-k)