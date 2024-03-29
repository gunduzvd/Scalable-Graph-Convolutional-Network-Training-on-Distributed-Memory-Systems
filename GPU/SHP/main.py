from math import floor
import os
import random
from re import sub
import kahypar as kahypar
import numpy as np
import scipy
import scipy.sparse as sparse
import itertools
import matplotlib.pylab as plt
from pathlib import Path
from itertools import accumulate
import pickle
import sys, getopt, os
import scipy.io

def partitionColNet(A, K):
    A = sparse.csc_matrix(A)
    num_nodes = A.shape[0]
    num_nets = A.shape[1]
    hyperedge_indices = A.indptr
    hyperedges = A.indices
    node_weights = [1] * num_nodes
    edge_weights = [1] * num_nets
    hypergraph = kahypar.Hypergraph(num_nodes, num_nets, hyperedge_indices, hyperedges, K, edge_weights, node_weights)
    context = kahypar.Context()
    context.loadINIconfiguration("km1_kKaHyPar_sea20.ini")
    context.setK(K)
    context.setEpsilon(0.03)
    kahypar.partition(hypergraph, context)
    partvect = [hypergraph.blockID(i) for i in hypergraph.nodes()]
    return partvect

def sample_submatrix(A, k):
    indices = set(range(0, A.shape[0]))
    subindices = sorted(random.sample(indices, A.shape[0] - k))
    S = A.copy()
    S[:, subindices] = 0
    indices = list(indices.difference(subindices))
    S = S[indices, :]
    S = S[~np.all(S == 0, axis=1)]
    return S

def sample_sparse_submatrix(A, k):
    indices = range(0, A.shape[0])
    subindices = np.array(random.sample(indices, k))
    
    filter = np.in1d(A.row, subindices)
    row = A.row[filter]
    col = A.col[filter]
    data = A.data[filter]

    filter = np.in1d(col, subindices)
    row = row[filter]
    col = col[filter]
    data = data[filter]

    sA = sparse.coo_matrix((data, (row, col)), shape=A.shape)
    sA = sparse.csc_matrix(sA)
    nonempty_cols = np.diff(sA.indptr) != 0
    sA = sA[:,nonempty_cols]
    return sA

def generate_stochastic_hypergraph(A, nbatches, batch_size):
    submatrices = []
    for i in range(nbatches):
        S = sample_sparse_submatrix(A, batch_size)
        submatrices.append(S)
    # H = sparse.vstack(submatrices)
    H = sparse.hstack(submatrices)
    # H = np.vstack(submatrices)
    return H

def compute_communication_volume(S, partvect):
    S = sparse.csc_matrix(S)
    comm_vol = 0
    for i in range(S.shape[1]):
        conn_set = set()
        for j in range(S.indptr[i], S.indptr[i+1]):
            conn_set.add(partvect[S.indices[j]])

        comm_vol += len(conn_set) - 1
    return comm_vol

def simulate(A, partvect_hp, partvect_stchp, niter, batch_size):
    # print(partvect_hp)
    # print(partvect_stchp)
    comm_vol_hp, comm_vol_stchp = 0, 0
    for i in range(niter):
        S = sample_sparse_submatrix(A, batch_size)
        comm_vol_hp += compute_communication_volume(S, partvect_hp)
        comm_vol_stchp += compute_communication_volume(S, partvect_stchp)
    print(comm_vol_hp, comm_vol_stchp)


def main(argv):
    global path, K, nsim_iter, batch_size, nbatches_for_stcH, output_dir
    try:
        opts, args = getopt.getopt(argv, "p:k:s:b:h:o:", [])
    except:
        print("i:f:")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-p':
            path = arg
        elif opt == '-k':
            K = int(arg)
        elif opt == '-s':
            nsim_iter = int(arg)
        elif opt == '-b':
            batch_size = int(arg)
        elif opt == '-h':
            nbatches_for_stcH = int(arg)
        elif opt == '-o':
            output_dir = arg

    np.set_printoptions(threshold=sys.maxsize)

    print(path)
    print(K, nsim_iter, batch_size, nbatches_for_stcH)

    A = scipy.io.mmread(path)
    # A.setdiag(1)

    partvect_hp = partitionColNet(A, K)
    stcH = generate_stochastic_hypergraph(A, nbatches_for_stcH, batch_size)
    partvect_stchp = partitionColNet(stcH, K)

    simulate(A, partvect_hp, partvect_stchp, nsim_iter, batch_size)

    import pickle
    name = f"partvec.hp.{K}"
    output_path_hp = os.path.join(output_dir, name)
    name = f"partvec.stchp.{K}"
    output_path_stchp = os.path.join(output_dir, name)
    print(output_path_hp, output_path_stchp)
    with open(output_path_hp, 'wb') as file:
        pickle.dump(partvect_hp, file)
    with open(output_path_stchp, 'wb') as file:
        pickle.dump(partvect_stchp, file)


# python main.py -p data/karate/karate.mtx -k 3 -s 1000 -b 10 -h 200
if __name__ == '__main__':
    main(sys.argv[1:])
    # A = np.array([[0,1,0,1,1,0], [1,0,1,1,0,1], [0,0,0,1,1,0], [1,0,1,0,1,0], [0,0,0,0,0, 1]])
    # A = sparse.coo_matrix(A)
    # sample_sparse_submatrix(A, 3)






############################################################################################
############################################################################################
############################################################################################

# S1 = sample_submatrix(A, 2)
# print(S1)
# S2 = sample_submatrix(A, 2)
# print(S2)
#
# M = sparse.vstack([S1, S2])
#
# print(M.todense())
#
# partvect = partitionColNet(M, 4)
# print(partvect)
