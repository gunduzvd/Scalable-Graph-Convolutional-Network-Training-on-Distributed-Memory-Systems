import sys, getopt, os
import time

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import Backend
import torch.multiprocessing as mp
from random import Random
from torch.utils.data import Dataset
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from math import ceil
import scipy
import scipy.sparse as sparse
import numpy as np
from scipy.io import mmread, mmwrite
from scipy.sparse import csr_matrix, coo_matrix, find, identity, lil_matrix, diags
import time

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

def compute_communication_maps(A, partvec, rank, size):
    global myrank
    send_map = {p: [] for p in range(size)}
    recv_map = {p: [] for p in range(size)}
    for i in range(A.nnz):
        if partvec[A.row[i]] == rank and partvec[A.col[i]] != rank:
            recv_map[partvec[A.col[i]]].append(A.col[i])
        if partvec[A.col[i]] == rank and partvec[A.row[i]] != rank:
            send_map[partvec[A.row[i]]].append(A.col[i])
    for p in range(size):
        send_map[p] = sorted(send_map[p])
        recv_map[p] = sorted(recv_map[p])
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
    return torch_coo_A.to_dense()

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

def communicate_fgm(H, backward = False):
    global send_map, recv_map, X, recv_buffers, cpu_device, cuda_device, recv_buffers, send_buffers

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

    for phase in range(2):
        comp = phase_comp(phase)
        for target in sorted(send.keys(), reverse=False):
            if comp(myrank, target):
                indices = send[target]
                temp_send_buffers[target] = H[indices]
                dist.send(tensor=temp_send_buffers[target], dst=target)

        for source in sorted(send.keys(), reverse=True):
            if not comp(myrank, source):
                indices = recv[source]
                dist.recv(temp_recv_buffers[source], src=source)
                X[indices] = temp_recv_buffers[source]

    H = H + X

    return H

class Comm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, H):
        H = communicate_fgm(H, backward=False)
        return H

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = communicate_fgm(grad_output, backward=True)
        return grad_output

class PGAT(nn.Module):
    def __init__(self, A, in_features, out_features):
        super(PGAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.send_map = send_map
        self.recv_map = recv_map
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.attention = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        nn.init.xavier_normal_(self.attention, gain=gain)

    def forward(self, H):
        Comm.apply(H)
        Z = self.linear(H)
        z1 = torch.matmul(Z, self.attention[:self.out_features, :])
        z2 = torch.matmul(Z, self.attention[self.out_features:, :])

        attention = z1 + z2.T
        zero_vec = torch.zeros(attention.shape)
        attention = torch.where(self.A > 0, attention, zero_vec)
        attention = F.softmax(attention, dim=1)

        H = torch.matmul(attention, Z)

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
    global myrank, world_size, send_map, recv_map, device, X, recv_buffers, cpu_device, cuda_device, recv_buffers, send_buffers
    myrank = rank
    world_size = size
    if backend == "gloo":
        device = torch.device(f'cpu:{myrank}')
    else:
        device = torch.device(f'cuda:{myrank % torch.cuda.device_count()}')

    # cpu_device = torch.device(f'cpu:{myrank}')
    # cuda_device = torch.device(f'cuda:{myrank}')
    # device = cpu_device

    # path_A = "A.txt"
    # path_partvec = "partvec.txt"
    # nlayers = 3

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

    x = [[e] * nfeatures for e in range(A.shape[0])]
    x = np.vstack(x)
    H = torch.tensor(x, dtype=torch.float32, requires_grad=True, device=device)
    X = torch.zeros(H.shape, device=device)

    # torch.autograd.set_detect_anomaly(True)
    labels = torch.arange(0, A.shape[0], device=device) % nfeatures

    model = nn.Sequential(
        *[PGAT(A, nfeatures, nfeatures) for _ in range(nlayers)]
    )

    # output = model(H)

    model = model.to(device)
    initiliaze_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    start = time.time()
    for epoch in range(50):

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
    if myrank == 0:
        print("Elapsed time {:.4f}".format(elapsed.item()))


def init_process(rank, size, fn, nlayers, nfeatures, path_A, path_partvec, backend):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, nlayers, nfeatures, path_A, path_partvec, backend)

def main(argv):
    global path_A, path_partvec, backend, size
    try:
        opts, args = getopt.getopt(argv, "a:p:b:s:l:f:", [])
    except:
        print("a:p:b:")
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

    nlayers = 1
    nfeatures = 4
    path_A = "A.txt"
    path_partvec = "partvec.txt"

    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, nlayers, nfeatures, path_A, path_partvec, backend))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main(sys.argv[1:])

