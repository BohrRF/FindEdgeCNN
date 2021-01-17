import numpy as np
import matplotlib.pyplot as plt
import batch_prepare
import fileprocess
import cnn
import pickle

dense_size = 2
length = 35
height = 35

test_degree = 45

alpha = fileprocess.read_alpha_shape("alpha_shape_points.csv")[test_degree]
alpha_graph = batch_prepare.single_alpha(alpha)

graphs = fileprocess.read_graph("relocated_points_all.csv")
test_graph, full_graph = batch_prepare.single_graph_add_window(length, height, graphs[test_degree])

model_path = "mnist_mlp_weights.h5"
#model_path = cnn.Result_path + "mnist_mlp_weights.h5"

model = cnn.build_model(dense_size, (35,35,1))
model.load_weights(model_path)

_result = model.predict(test_graph)
label = np.argmax(_result, axis=1)
result = label.reshape((110, 120))

fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharex="col", sharey="row")
axes[0].imshow(full_graph, cmap=plt.get_cmap("Blues"))
axes[0].set_title("Original Points")

axes[1].imshow(result, cmap=plt.get_cmap("plasma"))
axes[1].set_title("CNN Result")


"""ax = plt.axes([0.5, 0.6, 0.3, 0.6])"""

axes[2].imshow(alpha_graph, cmap=plt.get_cmap("plasma"))
axes[2].set_title("Alpha Shape")

plt.show()

