import sys, getopt, os
from scipy.sparse import csr_matrix, coo_matrix, find, identity, lil_matrix, diags
from scipy.io import mmread, mmwrite
import numpy as np

output_path_A, output_path_H, output_config_path, path = "", "", "", ""
nfeatures, nlayers = 3, 4
noutput_features = 2

def main(argv):
    global output_path_A, output_path_H, path, nfeatures, nlayers
    try:
        opts, args = getopt.getopt(argv, "i:f:l:", [])
    except:
        print("i:f:")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            path = arg
        elif opt == '-f':
            nfeatures = int(arg)
        elif opt == '-l':
            nlayers = int(arg)

if __name__ == '__main__':
    main(sys.argv[1:])

path_dir = os.path.dirname(path)
file_name = os.path.basename(path)
file_name = os.path.splitext(file_name)[0]

output_path_A = os.path.join(path_dir, file_name+ ".A")
output_path_H = os.path.join(path_dir, file_name+ ".H")
output_path_Y = os.path.join(path_dir, file_name+ ".Y")
output_config_path = os.path.join(path_dir, "config")
print(output_config_path)

print(output_path_A)
print(output_path_H)
print(path)
print(nfeatures)

A = mmread(path)

for i in range(A.nnz):
    if A.row[i] == A.col[i]:
        A.data[i] = 0
A.eliminate_zeros()

nvtx = A.shape[0]
I = identity(nvtx)
A = A + I

col_sum = A.sum(axis=0)
col_sum = np.asarray(col_sum).reshape(-1)
col_sum = np.sqrt(col_sum)
col_sum = 1/col_sum
#Dc = csr_matrix(np.diag(col_sum))
Dc = diags([col_sum], [0])

row_sum = A.sum(axis=1)
row_sum = np.asarray(row_sum.transpose()).reshape(-1)
row_sum = np.sqrt(row_sum)
row_sum = 1/row_sum
#Dr = csr_matrix(np.diag(row_sum))
Dr = diags([row_sum], [0])

A = Dr * A * Dc
#print(A.toarray())

#H = np.random.rand(nvtx, nfeatures)
H = np.ones(shape=(nvtx, nfeatures))
H = coo_matrix(H)

#Y = np.random.randint(2, size=(nvtx,1))
Y = np.ones(shape=(nvtx, 2))
Y[:,0] = 0
Y = coo_matrix(Y)

mmwrite(output_path_A, A, precision=3)
mmwrite(output_path_H, H, precision=1)
mmwrite(output_path_Y, Y, precision=1)

with open(output_config_path, "w") as f:
    arr = [nfeatures] * (nlayers)
    arr[-1] = noutput_features
    arr = " ".join(str(e) for e in arr)
    f.write(f"{nlayers} {nvtx} {arr}")




"""
A = lil_matrix((4, 4), dtype=np.float)
A[0,1] = 1
A[0,3] = 1
A[1,2] = 1
A[2,0] = 1
A[2,1] = 1
A[2,3] = 1
A[3,0] = 1
A = csr_matrix(A, dtype=np.float)

I = identity(4, dtype=np.float)
A = A + I
print(A.toarray())
"""



