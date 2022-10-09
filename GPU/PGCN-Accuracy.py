import sys, getopt, os
import time
import argparse
import sys

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Backend
import torch.multiprocessing as mp
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from random import Random
from math import ceil
import scipy
import scipy.sparse as sparse
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, coo_matrix, find, identity, lil_matrix, diags

world_size = 0
myrank = 0
send_map = None
recv_map = None
recv_buffers = None
send_buffers = None
device = None
path_A = None
path_partvec = None
X = None
cpu_device = None
cuda_device = None
stats = {}

batch_size = 10
nbatches = 5
batch_i = 0
batch_indices = []


def compute_communication_maps(A, partvec, rank, size):
    global myrank
    send_map = {p: set() for p in range(size)}
    recv_map = {p: set() for p in range(size)}
    for i in range(A.nnz):
        if partvec[A.row[i]] == rank and partvec[A.col[i]] != rank:
            recv_map[partvec[A.col[i]]].add(A.col[i])
        if partvec[A.col[i]] == rank and partvec[A.row[i]] != rank:
            send_map[partvec[A.row[i]]].add(A.col[i])
    for p in range(size):
        send_map[p] = torch.tensor(sorted(send_map[p]), dtype=torch.long, device=device)
        recv_map[p] = torch.tensor(sorted(recv_map[p]), dtype=torch.long, device=device)
    recv_map.pop(myrank)
    send_map.pop(myrank)

    return send_map, recv_map

def get_partitiont_of_adjacency_matrix(A, partvec, rank):
    global device, cpu_device, cuda_device
    indices = [index for (index, value) in enumerate(partvec) if value == rank]
    filter = np.in1d(A.row, indices)
    row = A.row[filter]
    col = A.col[filter]
    data = A.data[filter]
    coords = np.vstack((row, col))
    coords = torch.tensor(coords)
    vals = torch.tensor(data)
    torch_coo_A = torch.sparse_coo_tensor(coords, vals, A.shape, dtype=torch.float32, device=device)
    return torch_coo_A

def phase_comp(phase):
    def less(x,y):
        return x < y
    def greater(x,y):
        return x > y
    if phase == 0:
        return less
    elif phase == 1:
        return greater
    else:
        return None

def init_stats():
    global stats, device
    stats["send_volume"] = torch.tensor(0, device=device)
    stats["recv_volume"] = torch.tensor(0, device=device)
    stats["send_nmsg"] = torch.tensor(0, device=device)
    stats["recv_nmsg"] = torch.tensor(0, device=device)

def communicate_fgm(H, backward = False):
    global send_map, recv_map, X, recv_buffers, cpu_device, cuda_device, recv_buffers, \
        send_buffers, stats, batch_size, nbatches, batch_indices, batch_i

    if not backward:
        send = send_map
        recv = recv_map
        temp_recv_buffers = recv_buffers
        temp_send_buffers = send_buffers
    else:
        send = recv_map
        recv = send_map
        temp_recv_buffers = send_buffers
        temp_send_buffers = recv_buffers

    send_reqs, recv_reqs = [], {}
    for (target, indices) in send.items():
        if target == myrank:
            continue

        subindices = batch_indices[batch_i]
        combined = torch.cat((indices, subindices))
        uniques, counts = combined.unique(return_counts=True)
        intersection = uniques[counts > 1]

        h = H[intersection]
        req = dist.isend(tensor=h, dst=target)
        send_reqs.append(req)

    for (source, indices) in recv.items():
        if source == myrank:
            continue

        subindices = batch_indices[batch_i]
        combined = torch.cat((indices, subindices))
        uniques, counts = combined.unique(return_counts=True)
        intersection = uniques[counts > 1]

        h = torch.zeros(len(intersection), H.shape[1], device=device)
        dist.recv(h, src=source)
        X[intersection] = h

    for req in send_reqs:
        req.wait()

    H = H + X

    return H

class PSpMM(torch.autograd.Function):

    @staticmethod
    def forward(ctx, A, H):
        H = communicate_fgm(H, backward=False)
        ctx.save_for_backward(A)
        return torch.sparse.mm(A, H)

    @staticmethod
    def backward(ctx, grad_output):
        A, = ctx.saved_tensors
        grad_output = torch.sparse.mm(A.t(), grad_output)
        grad_output = communicate_fgm(grad_output, backward=True)
        return None, grad_output

class PGCN(nn.Module):
    def __init__(self, A, in_features, out_features):
        super(PGCN, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.A = A
        self.send_map = send_map
        self.recv_map = recv_map

    def forward(self, H):
        H = PSpMM.apply(self.A, H)
        H = self.linear(H)
        H = F.relu(H)
        return H

def average_gradients(model):
    global world_size
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= world_size

def initiliaze_parameters(model):
    global world_size
    for param in model.parameters():
        dist.all_reduce(param.data, op=dist.ReduceOp.SUM)
        param.data /= world_size

def run(rank, size, nlayers, nfeatures, path_A, path_partvec, backend):
    global myrank, world_size, send_map, recv_map, device, X, \
        recv_buffers, cpu_device, cuda_device, recv_buffers, send_buffers, stats, batch_size, nbatches, batch_indices, batch_i

    random_seed = 1
    np.random.seed(random_seed)
    random.seed(random_seed)

    myrank = rank
    world_size = size
    if backend == "gloo":
        device = torch.device(f'cpu:{myrank}')
    else:
        device = torch.device(f'cuda:{myrank % torch.cuda.device_count()}')

    A = mmread(path_A)
    with open(path_partvec) as f:
        partvec = list(map(int, f.readline().split()))

    send_map, recv_map = compute_communication_maps(A, partvec, rank, size)
    A = get_partitiont_of_adjacency_matrix(A, partvec, rank)

    send_buffers, recv_buffers = {}, {}
    for (source, indices) in recv_map.items():
        recv_buffers[source] = torch.zeros(len(indices), nfeatures, device=device)
    for (target,indices) in send_map.items():
        send_buffers[target] = torch.zeros(len(indices), nfeatures, device=device)

    init_stats()

    x = [[e] * nfeatures for e in range(A.shape[0])]
    x = np.vstack(x)
    H = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
    X = torch.zeros(H.shape, device=device)

    # torch.autograd.set_detect_anomaly(True)
    labels = torch.arange(0, A.shape[0], device=device) % nfeatures

    model = nn.Sequential(
        *[PGCN(A,nfeatures,nfeatures) for _ in range(nlayers)]
    )

    model = model.to(device)
    initiliaze_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    batch_size = 256
    nbatches = 5
    for _ in range(nbatches):
        indices = set(range(0, H.size(0)))
        subindices = np.array(random.sample(indices, batch_size))
        subindices = torch.tensor(subindices, dtype=torch.long, device=device)
        batch_indices.append(subindices)

    start = time.time()
    for epoch in range(15):
        for batch_i in range(nbatches):
            logits = model(H)

            logp = F.log_softmax(logits, 1)
            loss = F.nll_loss(logp, labels)

            optimizer.zero_grad()
            loss.backward()
            average_gradients(model)
            optimizer.step()

            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            if myrank == 0:
                print("Epoch {:05d} | Loss {:.4f}".format(epoch, loss))

    elapsed = time.time() - start
    elapsed = torch.tensor([elapsed], device=device)
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)

    # print(stats)
    total_vol = stats["send_volume"]
    total_nmsg = stats["send_nmsg"]
    dist.all_reduce(total_vol, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_nmsg, op=dist.ReduceOp.SUM)

    if myrank == 0:
        print("Elapsed time {:.4f}".format(elapsed.item()), flush=True)
        print(f"total_vol: {total_vol} total_nmsg: {total_nmsg}")


def init_process(rank, size, fn, nlayers, nfeatures, path_A, path_partvec, backend):
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, nlayers, nfeatures, path_A, path_partvec, backend)

def main(argv):
    global path_A, path_partvec, backend, size, rank
    size = int(os.environ["SLURM_NPROCS"])
    rank = int(os.environ["SLURM_PROCID"])
    os.environ["RANK"] = os.environ["SLURM_PROCID"]
    try:
        opts, args = getopt.getopt(argv, "a:p:b:s:l:f:", [])
    except:
        print("a:p:b:", flush=True)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-a':
            path_A = arg
        elif opt == '-p':
            path_partvec = arg
        elif opt == '-b':
            backend = arg
        elif opt == '-s':
            size = int(arg)
        elif opt == '-l':
            nlayers = int(arg)
        elif opt == '-f':
            nfeatures = int(arg)

    mp.set_start_method("spawn")
    p = mp.Process(target=init_process, args=(rank, size, run, nlayers, nfeatures, path_A, path_partvec, backend))
    p.start()
    p.join()

if __name__ == '__main__':
    main(sys.argv[1:])

