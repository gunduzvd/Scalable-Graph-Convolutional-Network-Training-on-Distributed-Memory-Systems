# Data Preprocessing

Before running partitioning codes, data preprocessing is applied to normalize the adjacency matrix and to produce config files indicating how many features and layers are needed.

python GrB-GNN-IDG.py -i {PATH-TO-FILE} -f {nfeatures} -l {nlayers}

{PATH-TO-FILE} should point to a graph in matrix market format. Datasets in the paper are available through https://sparse.tamu.edu/

-f {nfeatures} : number of features per layer

-l {nfeatures} : number of layers

This code produces. *.A.mtx, *.H.mtx, *.Y.mtx and config files.

# DGL implementation of GCN

To run DGL implementataion

python gcn.py -a {FILE_A} -h {FILE_H} -y {FILE_Y} -c {FILE_C}

-a {FILE_A} : path to *.A.mtx

-h {FILE_H} : path to *.H.mtx

-y {FILE_Y} : path to *.Y.mtx

-c {FILE_C} : path to config file

# CAGNET code for parallel GCN 

The code requires SuiteSparse:GraphBLAS (https://github.com/DrTimothyAldenDavis/GraphBLAS) and MPI libraries. Modify INC_DIR and LIB_DIR variables in the makefile to point correct locations. It only performs inference phase.

To compile the code just run the command:

make

To run the executable:

mpirun -n {nprocs} cagnet1d -p {DATA_PATH} -c {CONFIG_PATH} -t {NTHREADS}

Parameters:

-n {nprocs} : number of MPI processes

-p {DATA_PATH} : path to folde that contains

-c {CONFIG_PATH} : *.A.mtx, *.H.mtx, *.Y.mtx and config files.

-t {NTHREADS} : number of threads per MPI process


# Graph(GCN-GP) and Hypergraph(GCN-HP) Partitioning Codes

The input matrix partitioning code for parallel GCN training algorithm. The code uses patoh and metis partitioning libraries.
Modify INC_DIR and LIB_DIR to point appropriate locations in makefile.

To compile the partitioning code just use the command:

make

To run Hypergraph Partitioning:

./gcnhgp -a {Par_A} -h {Par_H} -o {Par_O} -k {k} -f {nfeatures} -l {nlayers}  

To run Graph Partitioning:

./gcngp -a {Par_A} -h {Par_H} -o {Par_O} -k {k} -f {nfeatures} -l {nlayers} 

Parameters:

-a {Par_A} : path to adjacecny matrix 

-h {Par_H} : path to input vertex features 

-o {Par_O} : output folder for partitioned matrices 

-k {k} : Number of partitions 

-f {nfeatures} : number of features per layer 

-l {nlayers} : number of layers 

# Scalable Graph Convolutional Network Training on Distributed-Memory Systems

The code requires SuiteSparse:GraphBLAS (https://github.com/DrTimothyAldenDavis/GraphBLAS) and MPI libraries. Modify INC_DIR and LIB_DIR variables in the makefile to point correct locations.

To compile the code just run the command:

make

To run the executable:

mpirun -n {nprocs} grbgcn -p {DATA_PATH} -c {CONFIG_PATH} -t {NTHREADS}

Parameters:

-n {nprocs} : number of MPI processes

-p {DATA_PATH} : path to folde that contains

-c {CONFIG_PATH} : *.A.mtx, *.H.mtx, *.Y.mtx and config files.

-t {NTHREADS} : number of threads per MPI process


# Random Hypergraph Model

Code for random hypergraph partitioning model. The code simulates mini-bathes and compares random hypergraph model against the baseline. It uses https://github.com/kahypar/kahypar hypergraph partitioning tool which is a good alternative for patoh. To install the dependency use "pip install kahypar==1.1.7"

To run the code

python main.py -p data/com-Amazon/com-Amazon.mtx -k {nparts} -s {nsimulations} -b {batch-szie} -h {nbatches-for-model}

-k {nparts} : Number of partitions

-s {nsimulations} : Number of mini-batches for simulation

-b {batch-szie} : mini-batch size interms of vertices

-h {nbatches-for-model} : number of mini-batches to build random hypergraph model


# GPU implementation

GPU folder stores PyTorch implementation (with NCCL backend) of the proposed parallel training algorithm.

PGCN.py is the GPU version of the proposed parallel GCN training algorithm. sample execution is as follows:

$ python PGCN.py -a <path to adj matrix in matrix market format> -p <path to part vector> -b <nccl|gloo> -s <# of gpus> -l <~ of layers> -f <dimension of hidden layers>

PGCN-Accuracy.py is for experiments performed on cora dataset to see if the proposed algorithm affects predictive performance.

PGCN-Mini-batch.py achieves mini-batch training instead of full-batch training.

PGAT.py is a sample implementation that demonstrates how graph attention networks can be supported by the proposed partitioning and training algorithm.

pytorch.3node.slurm is sample script that shows how to run the codes by slurm.




