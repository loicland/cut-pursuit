import libcp
obs = [[1000.],[0.],[0.]]
source = [[0, 1, 1, 2]]
target = [[1, 2, 0, 1]]
libcp.cutpursuit(obs, source, target, 0.01)
