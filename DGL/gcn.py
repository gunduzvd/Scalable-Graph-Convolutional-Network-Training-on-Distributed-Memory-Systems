import dgl
import sys, getopt
from scipy.sparse import csr_matrix, coo_matrix, find, identity, lil_matrix
from scipy.io import mmread, mmwrite
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv
import sys, getopt
import time

class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, x):
        with g.local_scope():
            x = dgl.ops.copy_u_sum(g, x)
            x = self.linear(x)
            return x

path_to_A, path_to_H, path_to_Y, path_to_config = "", "", "", ""

def main(argv):
    global path_to_A, path_to_H, path_to_Y, path_to_config
    try:
        opts, args = getopt.getopt(argv, "a:h:c:y:", [])
    except:
        print("a:h:c:")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-a':
            path_to_A = arg
        elif opt == '-h':
            path_to_H = arg
        elif opt == '-y':
            path_to_Y = arg
        elif opt == '-c':
            path_to_config = arg

if __name__ == '__main__':
    main(sys.argv[1:])

with open(path_to_config) as f:
    line = list(map(int, f.readline().split()))
    dims = line[2:]
    nlayers = line[0]
print(dims)
print(nlayers)

A = mmread(path_to_A)
G = dgl.graph((A.row, A.col))

#H = np.random.rand(A.shape[0], dims[0])
#H = mmread(path_to_H)
H = np.ones(shape=[A.shape[0], dims[0]])
H = th.tensor(H, dtype=th.float32)

Y = mmread(path_to_Y)
Y = np.squeeze(Y.toarray()[:,1])
#Y = np.ones(shape=(A.shape[0],))
labels = th.tensor(Y, dtype=th.long)

class GCN(nn.Module):
    def __init__(self, dims):
        super(GCN, self).__init__()
        self.nlayers = len(dims) - 1
        self.layers = nn.ModuleList([GCNLayer(dims[i], dims[i + 1]) for i in range(self.nlayers)])

    def forward(self, g, inputs):
        h = inputs
        for i in range(self.nlayers):
            h = self.layers[i](g, h)
            h = th.sigmoid(h)
        return h

net = GCN(dims)

print("ca:",th.cuda.is_available())
print(G.device)

optimizer = th.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t = time.process_time()
for epoch in range(5):
    logits = net(G, H)
    loss = F.cross_entropy(logits, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d | Loss: %.4f' % (epoch, loss.item()))
elapsed_time = time.process_time() - t
print("Time elapsed: ", elapsed_time)
