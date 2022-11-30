#Let's draw our teaining and predicting datapoints
#This part of the code also can be found in qiskit -> Quantum Kernel Machine Learning webpage.


plt.figure(figsize=(5, 5))
plt.ylim(0, 2 * np.pi)
plt.xlim(0, 2 * np.pi)
plt.imshow(
    np.asmatrix(adhoc_total).T,
    interpolation="nearest",
    origin="lower",
    cmap="RdBu",
    extent=[0, 2 * np.pi, 0, 2 * np.pi],
)

plt.scatter(
    train_features[np.where(train_labels[:] == 0), 0],
    train_features[np.where(train_labels[:] == 0), 1],
    marker="s",
    facecolors="w",
    edgecolors="b",
    label="A train",
)
plt.scatter(
    train_features[np.where(train_labels[:] == 1), 0],
    train_features[np.where(train_labels[:] == 1), 1],
    marker="o",
    facecolors="w",
    edgecolors="r",
    label="B train",
)
plt.scatter(
    test_features[np.where(Y_predict[:] == 0), 0],
    test_features[np.where(Y_predict[:] == 0), 1],
    marker="s",
    facecolors="b",
    edgecolors="w",
    label="A test",
)
plt.scatter(
    test_features[np.where(Y_predict[:] == 1), 0],
    test_features[np.where(Y_predict[:] == 1), 1],
    marker="o",
    facecolors="r",
    edgecolors="w",
    label="B test",
)

plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
plt.title("Ad hoc dataset for classification")

plt.show()


#Let's visualize kernel matrix (it is called k in the kernel preparation notebook.)
plt.figure(figsize=(5, 5))
plt.imshow(np.asmatrix(k),interpolation='nearest', origin='upper')
plt.title("Analytical Kernel Matrix")
plt.colorbar()
plt.show()