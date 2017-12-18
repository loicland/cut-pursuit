import libcp
obs = np.array([[1000.],[0.],[0.]], dtype = 'float32')
source = np.array([[0, 1, 1, 2]], dtype = 'uint32')
target = np.array([[1, 2, 0, 1]], dtype = 'uint32')
edge_weight = np.array([[1., 1., 1., 1.]], dtype = 'float32')
libcp.cutpursuit(obs, source, target, edge_weight, 0.01)
