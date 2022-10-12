# Data Preprocessing {./preprocess}

Before running partitioning codes, data preprocessing is applied to normalize the adjacency matrix and to produce config files indicating how many features and layers are needed.

```
python GrB-GNN-IDG.py -i {PATH-TO-FILE} -f {nfeatures} -l {nlayers}
```

Parameters:

{PATH-TO-FILE} should point to a graph in matrix market format. Datasets in the paper are available through https://sparse.tamu.edu/

-f {nfeatures} : number of features per layer

-l {nlayers} : number of layers

This code produces. *.A.mtx, *.H.mtx, *.Y.mtx and config files.

# DGL implementation of GCN {./DGL}

To run DGL implementation

```
python gcn.py -a {FILE_A} -h {FILE_H} -y {FILE_Y} -c {FILE_C}
```

Parameters:

-a {FILE_A} : path to *.A.mtx

-h {FILE_H} : path to *.H.mtx

-y {FILE_Y} : path to *.Y.mtx

-c {FILE_C} : path to config file

# CAGNET code for parallel GCN {./Cagnet}

The code requires SuiteSparse:GraphBLAS (https://github.com/DrTimothyAldenDavis/GraphBLAS) and MPI libraries. Modify INC_DIR and LIB_DIR variables in the makefile to point correct locations. It only performs inference phase.

To compile the code just run the command:

```
make
```

To run the executable:

```
mpirun -n {nprocs} cagnet1d -p {DATA_PATH} -c {CONFIG_PATH} -t {nthreads}
```

Parameters:

-n {nprocs} : number of MPI processes

-p {DATA_PATH} : path to folder that contains *.A.mtx, *.H.mtx, *.Y.mtx

-c {CONFIG_PATH} : path to config file

-t {nthreads} : number of threads per MPI process


# Graph {./GCN-GP} and Hypergraph {./GCN-HP} Partitioning Codes 

The input matrix partitioning code for parallel GCN training algorithm. The code uses patoh and metis partitioning libraries.
Modify INC_DIR and LIB_DIR to point appropriate locations in makefile.

To compile the partitioning code just use the command:

```
make
```

To run Hypergraph Partitioning:

```
./gcnhgp -a {Par_A} -h {Par_H} -o {Par_O} -k {k} -f {nfeatures} -l {nlayers}  
```

To run Graph Partitioning:

```
./gcngp -a {Par_A} -h {Par_H} -o {Par_O} -k {k} -f {nfeatures} -l {nlayers} 
```

Parameters:

-a {Par_A} : path to adjacency matrix 

-h {Par_H} : path to input vertex features 

-o {Par_O} : output folder for partitioned matrices 

-k {k} : number of partitions 

-f {nfeatures} : number of features per layer 

-l {nlayers} : number of layers 

# Scalable Graph Convolutional Network Training on Distributed-Memory Systems {./Parallel-GCN}

The code requires SuiteSparse:GraphBLAS (https://github.com/DrTimothyAldenDavis/GraphBLAS) and MPI libraries. Modify INC_DIR and LIB_DIR variables in the makefile to point correct locations.

To compile the code just run the command:

```
make
```

To run the executable:

```
mpirun -n {nprocs} grbgcn -p {DATA_PATH} -c {CONFIG_PATH} -t {nthreads}
```

Parameters:

-n {nprocs} : number of MPI processes

-p {DATA_PATH} : path to folde that contains

-c {CONFIG_PATH} : *.A.mtx, *.H.mtx, *.Y.mtx and config files.

-t {nthreads} : number of threads per MPI process


# Stochastic Hypergraph Model {./RHP}

Code for random hypergraph partitioning model. The code simulates mini-bathes and compares random hypergraph model against the baseline. It uses https://github.com/kahypar/kahypar hypergraph partitioning tool which is a good alternative for patoh. 

To install the dependency use 
```
pip install kahypar==1.1.7
```

To run the code

```
python main.py -p {path_a} -k {nparts} -s {nsimulations} -b {batch-size} -h {nbatches-for-model}
```

Parameters:

-p {path_a}: Path to adj matrix in matrix market format

-k {nparts} : Number of partitions

-s {nsimulations} : Number of mini-batches for simulation

-b {batch-size} : mini-batch size interms of vertices

-h {nbatches-for-model} : number of mini-batches to build random hypergraph model


# GPU implementation {./GPU}

GPU folder stores PyTorch implementation (with NCCL backend) of the proposed parallel training algorithm.

PGCN.py is the GPU version of the proposed parallel GCN training algorithm. Sample execution is as follows:

```
python PGCN.py -a {PATH_A} -p {PATH_P} -b {"nccl"|"gloo"} -s {numgpu} -l {numlayers} -f {hiddendim}
```

Parameters:

-a {PATH_A} : path to adj matrix in matrix market format

-p {PATH_P} : path to part vector

-b {"nccl"|"gloo"} : communication backend

-s {numgpu} : number of GPUs

-l {numlayers} : number of layers

-f {hiddendim} : dimension of hidden layers


PGCN-Accuracy.py is for experiments performed on cora dataset to see if the proposed algorithm affects predictive performance.

PGCN-Mini-batch.py achieves mini-batch training instead of full-batch training.

PGAT.py is a sample implementation that demonstrates how graph attention networks can be supported by the proposed partitioning and training algorithm.

pytorch.3node.slurm is sample script that shows how to run the codes by slurm. It can be used as follows
```
sbatch pytorch.3node.slurm -d {dataset} -l {numlayers} -f {hiddendim} -p {"hp"|"gp"|"rp"}
```
