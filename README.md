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





