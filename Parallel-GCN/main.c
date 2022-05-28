#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include <omp.h>
#include <math.h>
#include <malloc.h>
#include <getopt.h>

#include "GraphBLAS.h"
#include "mpi.h"

int npes, myrank;

int nlayers;
int *nneurons;
float alpha = 0.01;

int64_t *send_buffer_sizes, *recv_buffer_sizes;
int64_t max_buffer_size = 0;


GrB_Matrix A;
GrB_Matrix Y;
GrB_Matrix * H, *AH, *AG, * W, *Hcap, *Gcap, *Z, *G, * dW;
GrB_Matrix * Hsend;
int * Hrecv;
int nrecvs = 0;
int max_nfeatures = 50;

char path[256], config_path[256];
int nthreads;

void read_matrix(GrB_Matrix * M_ptr, char * path, int type);
void read_matrix2(GrB_Matrix * M_ptr, char * path);
void read_config(char * path);
void read_buffer_sizes(char * path);
void initialize_matrices();
void read_connectivity(char * path);
void statistics();
void GCN();

GrB_Info info;

#define ERROR(message,info)                                         \
{                                                                           \
    fprintf (stderr, "Graphblas error: %s\n[%d]\nFile: %s Line: %d\n",        \
        message, info, __FILE__, __LINE__) ;                                \
}

#define OK(method)                                                  \
{                                                                           \
    info = method ;                                                         \
    if (! (info == GrB_SUCCESS || info == GrB_NO_VALUE))                    \
    {                                                                       \
        ERROR ("", info) ;                                          \
    }                                                                       \
}

enum STATS {
    recv_comm_volume, send_comm_volume, recv_message_count, send_message_count
};
long long int stats[20];

void max_fp32(void *z, const void *x) {
    (*(float *) z) = fminf((*(float *) x), (float) 1.0);
}

void binary_cross_entropy_loss(void *z, const void *x, const void *y) {
    //(*(float *) z) = (*(float *) y) * log((*(float *) x)) + (1 - *(float *) y) * log(1 - (*(float *) x));
    (*(float *) z) = -1 * (*(float *) y) * log((*(float *) x));
}

void gradient_update(void *z, const void *x, const void *y) {
    (*(float *) z) = (*(float *) x) - alpha * (*(float *) y);
}

void sigmoid(void *z, const void *x) {
    (*(float *) z) = 1 / (1 + expf(-*(float *) x));
}

void sigmoid_derivative(void *z, const void *x) {
    sigmoid(z, x);
    (*(float *) z) = (*(float *) z) * (1 - (*(float *) z));
}

void cross_entropy_derivative_divisor(void *z, const void *x) {
    (*(float *) z) = (*(float *) x) * (1 - (*(float *) x));
}

float get_random(float x, float y) {
    float r = (float) rand() / (float) RAND_MAX;
    return x + (y - x) * r;
}

int c;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

    while ((c = getopt(argc, argv, "p:t:c:")) != -1) {
        switch (c) {
            case 'p':
                strcpy(path, optarg);
                if (path[strlen(path) - 1] != '/') {
                    strcat(path, "/");
                }
                break;
            case 'c':
                strcpy(config_path, optarg);
                break;
            case 't':
                nthreads = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            default:
                return 1;
                abort();
        }
    }

    //GrB_init(GrB_NONBLOCKING);
    //GrB_init(GrB_BLOCKING);
    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_TOP_PAD, 16 * 1024 * 1024);
    GxB_init(GrB_NONBLOCKING, malloc, calloc, realloc, free, true);
    GxB_set(GxB_NTHREADS, nthreads);

    char path_A[512];
    char path_H[512];
    char path_Y[512];
    char path_conf[512];
    char path_conn[512];
    char path_buff[512];
    sprintf(path_A, "%sA.%d", path, myrank);
    sprintf(path_H, "%sH.%d", path, myrank);
    sprintf(path_Y, "%sY.%d", path, myrank);
    sprintf(path_conf, "%s%s", path, "config");
    sprintf(path_conn, "%sconn.%d", path, myrank);
    sprintf(path_buff, "%sbuff.%d", path, myrank);

    read_config(path_conf);
    initialize_matrices();
    read_buffer_sizes(path_buff);
    read_matrix(&A, path_A, 0);
    //read_matrix(&H[0], path_H, 1);
    read_matrix2(&H[0], path_H);
    read_matrix(&Y, path_Y, 2);
    read_connectivity(path_conn);

    GCN();
    statistics();

    GrB_finalize();
    MPI_Finalize();

    return 0;
}

void GCN() {
    char ** send_buffer = (char **) malloc(npes * sizeof (char *));
    char ** recv_buffer = (char **) malloc(npes * sizeof (char *));

    for (int i = 0; i < npes; i++) {
        if (myrank == i) {
            send_buffer[i] = NULL;
            recv_buffer[i] = NULL;
            continue;
        }

        if (max_buffer_size <= send_buffer_sizes[i]) {
            max_buffer_size = send_buffer_sizes[i];
        }

        if (send_buffer_sizes[i] != 0) {
            send_buffer_sizes[i] = (3 * max_nfeatures * send_buffer_sizes[i] + 1) * sizeof (GrB_Index);
            send_buffer[i] = (char *) malloc(send_buffer_sizes[i]);

        }

        if (recv_buffer_sizes[i] != 0) {
            recv_buffer_sizes[i] = (3 * max_nfeatures * recv_buffer_sizes[i] + 1) * sizeof (GrB_Index);
            recv_buffer[i] = (char *) malloc(recv_buffer_sizes[i]);
        }

    }

    if (max_buffer_size < max_nfeatures)
        max_buffer_size = max_nfeatures;

    GrB_Index *I = (GrB_Index *) malloc(max_nfeatures * max_buffer_size * sizeof (GrB_Index));
    GrB_Index *J = (GrB_Index *) malloc(max_nfeatures * max_buffer_size * sizeof (GrB_Index));
    float * X = (float *) malloc(max_nfeatures * max_buffer_size * sizeof (float));
    float * dw_recv_buffer = (float *) malloc(max_nfeatures * max_nfeatures * sizeof (float));

    GrB_Index nvals;

    MPI_Request send_reqs[npes], recv_reqs[nrecvs];
    MPI_Status send_stats[npes], recv_stats[nrecvs];

    GrB_UnaryOp sigmoid_op = NULL, sigmoid_derivative_op = NULL,
            cross_entropy_derivative_divisor_op = NULL;
    GrB_UnaryOp_new(&sigmoid_op, sigmoid, GrB_FP32, GrB_FP32);
    GrB_UnaryOp_new(&sigmoid_derivative_op, sigmoid_derivative, GrB_FP32, GrB_FP32);
    GrB_UnaryOp_new(&cross_entropy_derivative_divisor_op, cross_entropy_derivative_divisor, GrB_FP32, GrB_FP32);

    GrB_BinaryOp gradient_update_op = NULL, binary_cross_entropy_loss_op = NULL;
    GrB_BinaryOp_new(&gradient_update_op, gradient_update, GrB_FP32, GrB_FP32, GrB_FP32);
    GrB_BinaryOp_new(&binary_cross_entropy_loss_op, binary_cross_entropy_loss, GrB_FP32, GrB_FP32, GrB_FP32);
    GrB_BinaryOp_new(&binary_cross_entropy_loss_op, binary_cross_entropy_loss, GrB_FP32, GrB_FP32, GrB_FP32);

    GrB_Monoid monoid;
    GrB_Monoid_new_FP32(&monoid, GrB_PLUS_FP32, true);

    stats[recv_comm_volume] = 0;
    stats[send_comm_volume] = 0;
    stats[recv_message_count] = 0;
    stats[send_message_count] = 0;

    GrB_Matrix T = NULL;
    GrB_Matrix_new(&T, GrB_FP32, nneurons[0], nneurons[nlayers]);

    double tic, toc;
    tic = MPI_Wtime();
    for (int epoch = 0; epoch < 3; epoch++) {

        for (int layer = 1; layer < nlayers; layer++) {

            int MID = layer;

            int count = 0;
            for (int i = 0; i < npes; i++) {
                if (Hrecv[i] != -1) {
                    MPI_Irecv(recv_buffer[i], recv_buffer_sizes[i], MPI_PACKED, i, MID, MPI_COMM_WORLD, &(recv_reqs[count]));
                    count++;
                }
            }

            for (int i = 0; i < npes; i++) {
                send_reqs[i] = MPI_REQUEST_NULL;
                if (Hsend[i] != NULL) {

                    OK(GrB_Matrix_clear(Hcap[layer - 1]));
                    OK(GrB_mxm(Hcap[layer - 1], NULL, NULL, GxB_PLUS_SECOND_FP32,
                            Hsend[i], H[layer - 1], GrB_NULL));

                    OK(GrB_Matrix_nvals(&nvals, Hcap[layer - 1]));
                    OK(GrB_Matrix_extractTuples(I, J, X, &nvals, Hcap[layer - 1]));

                    int position = 0;
                    MPI_Pack(&nvals, 1, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(I, nvals, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(J, nvals, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(X, nvals, MPI_FLOAT, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);

                    MPI_Isend(send_buffer[i], position, MPI_PACKED, i, MID, MPI_COMM_WORLD, &send_reqs[i]);

                    stats[send_comm_volume] += nvals;
                    stats[send_message_count]++;
                }
            }

            OK(GrB_Matrix_new(&(AH[layer - 1]), GrB_FP32, nneurons[0], nneurons[layer]));

            OK(GrB_mxm(AH[layer - 1], NULL, NULL, GxB_PLUS_TIMES_FP32,
                    A, H[layer - 1], GrB_DESC_R));

            count = 0;
            while (count < nrecvs) {
                int index;
                MPI_Status status;
                MPI_Waitany(nrecvs, recv_reqs, &index, &status);

                int position = 0;
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, &nvals, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, I, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, J, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, X, nvals, MPI_FLOAT, MPI_COMM_WORLD);
                stats[recv_comm_volume] += nvals;
                stats[recv_message_count]++;

                OK(GrB_Matrix_clear(Hcap[layer - 1]));
                OK(GrB_Matrix_build(Hcap[layer - 1], I, J, X, nvals, GrB_PLUS_FP32));

                OK(GrB_mxm(AH[layer - 1], NULL, GrB_PLUS_FP32, GxB_PLUS_TIMES_FP32,
                        A, Hcap[layer - 1], GrB_NULL));

                count++;
            }

            OK(GrB_Matrix_new(&(Z[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]));

            OK(GrB_mxm(Z[layer], NULL, NULL, GxB_PLUS_TIMES_FP32,
                    AH[layer - 1], W[layer], GrB_NULL));

            OK(GrB_Matrix_free(&(AH[layer - 1])));

            OK(GrB_apply(H[layer], NULL, NULL, sigmoid_op, Z[layer], GrB_DESC_R));

            for (int i = 0; i < npes; i++) {
                if (send_reqs[i] != MPI_REQUEST_NULL) {
                    MPI_Wait(&send_reqs[i], &send_stats[i]);
                }
            }

        }

        OK(GrB_eWiseAdd(T, NULL, NULL, binary_cross_entropy_loss_op, H[nlayers - 1], Y, GrB_DESC_R));
        float err, t_err;
        OK(GrB_reduce(&err, NULL, monoid, T, NULL));
        MPI_Reduce(&err, &t_err, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (myrank == 0)
            printf("err:%g\n", t_err);

        OK(GrB_apply(T, NULL, NULL, cross_entropy_derivative_divisor_op, H[nlayers - 1], GrB_DESC_R));

        OK(GrB_eWiseAdd(H[nlayers - 1], NULL, NULL, GrB_MINUS_FP32, H[nlayers - 1], Y, GrB_DESC_R));
        OK(GrB_eWiseMult(H[nlayers - 1], NULL, NULL, GrB_DIV_FP32, H[nlayers - 1], T, GrB_DESC_R));

        OK(GrB_apply(Z[nlayers - 1], NULL, NULL, sigmoid_derivative_op, Z[nlayers - 1], GrB_DESC_R));
        OK(GrB_eWiseMult(G[nlayers - 1], NULL, NULL, GrB_TIMES_FP32, H[nlayers - 1], Z[nlayers - 1], GrB_DESC_R));

        OK(GrB_Matrix_free(&(Z[nlayers - 1])));

        OK(GrB_Matrix_apply_BinaryOp2nd_FP32(G[nlayers - 1], NULL, NULL, GrB_DIV_FP32, G[nlayers - 1], nneurons[0], GrB_DESC_R));

        //MPI_Barrier(MPI_COMM_WORLD);
        for (int layer = nlayers - 1; layer > 0; layer--) {

            int MID = layer;

            int count = 0;
            for (int i = 0; i < npes; i++) {
                if (Hrecv[i] != -1) {
                    MPI_Irecv(recv_buffer[i], recv_buffer_sizes[i], MPI_PACKED, i, MID, MPI_COMM_WORLD, &(recv_reqs[count]));
                    count++;
                }
            }

            for (int i = 0; i < npes; i++) {
                send_reqs[i] = MPI_REQUEST_NULL;
                if (Hsend[i] != NULL) {

                    OK(GrB_Matrix_clear(Gcap[layer]));
                    OK(GrB_mxm(Gcap[layer], NULL, NULL, GxB_PLUS_SECOND_FP32,
                            Hsend[i], G[layer], GrB_NULL));

                    OK(GrB_Matrix_nvals(&nvals, Gcap[layer]));
                    OK(GrB_Matrix_extractTuples(I, J, X, &nvals, Gcap[layer]));

                    int position = 0;
                    MPI_Pack(&nvals, 1, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(I, nvals, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(J, nvals, MPI_UNSIGNED_LONG, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);
                    MPI_Pack(X, nvals, MPI_FLOAT, send_buffer[i], send_buffer_sizes[i], &position, MPI_COMM_WORLD);

                    MPI_Isend(send_buffer[i], position, MPI_PACKED, i, MID, MPI_COMM_WORLD, &send_reqs[i]);

                    stats[send_comm_volume] += nvals;
                    stats[send_message_count]++;
                }
            }

            OK(GrB_Matrix_new(&(AG[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]));

            OK(GrB_mxm(AG[layer], NULL, NULL, GxB_PLUS_TIMES_FP32,
                    A, G[layer], GrB_DESC_R));

            count = 0;
            while (count < nrecvs) {
                int index;
                MPI_Status status;
                MPI_Waitany(nrecvs, recv_reqs, &index, &status);

                int position = 0;
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, &nvals, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, I, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, J, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                MPI_Unpack(recv_buffer[status.MPI_SOURCE], recv_buffer_sizes[status.MPI_SOURCE],
                        &position, X, nvals, MPI_FLOAT, MPI_COMM_WORLD);
                stats[recv_comm_volume] += nvals;
                stats[recv_message_count]++;

                OK(GrB_Matrix_clear(Gcap[layer]));
                OK(GrB_Matrix_build(Gcap[layer], I, J, X, nvals, GrB_PLUS_FP32));

                OK(GrB_mxm(AG[layer], NULL, GrB_PLUS_FP32, GxB_PLUS_TIMES_FP32,
                        A, Gcap[layer], GrB_NULL));

                count++;
            }

            if (layer != 1) {
                OK(GrB_mxm(G[layer - 1], NULL, NULL, GxB_PLUS_TIMES_FP32,
                        AG[layer], W[layer], GrB_DESC_RT1));
                OK(GrB_apply(Z[layer - 1], NULL, NULL, sigmoid_derivative_op, Z[layer - 1], GrB_DESC_R));
                OK(GrB_eWiseMult(G[layer - 1], NULL, NULL, GrB_TIMES_FP32, G[layer - 1], Z[layer - 1], GrB_DESC_R));
            }

            OK(GrB_Matrix_free(&(Z[layer - 1])));

            GrB_Matrix_new(&(dW[layer]), GrB_FP32, nneurons[layer], nneurons[layer + 1]);
            OK(GrB_Matrix_assign_FP32(dW[layer], NULL, NULL, 0, GrB_ALL, 0, GrB_ALL, 0, GrB_NULL));
            OK(GrB_mxm(dW[layer], NULL, GrB_PLUS_FP32, GxB_PLUS_TIMES_FP32,
                    H[layer - 1], AG[layer], GrB_DESC_T0));

            OK(GrB_Matrix_free(&(AG[layer])));

            OK(GrB_Matrix_nvals(&nvals, dW[layer]));
            OK(GrB_Matrix_extractTuples(I, J, X, &nvals, dW[layer]));

            MPI_Allreduce(X, dw_recv_buffer, nvals, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

            OK(GrB_Matrix_clear(dW[layer]));
            OK(GrB_Matrix_build(dW[layer], I, J, dw_recv_buffer, nvals, GrB_PLUS_FP32));

            OK(GrB_eWiseAdd(W[layer], NULL, NULL, gradient_update_op, W[layer], dW[layer], GrB_DESC_R));
            OK(GrB_Matrix_free(&(dW[layer])));

            for (int i = 0; i < npes; i++) {
                if (send_reqs[i] != MPI_REQUEST_NULL) {
                    MPI_Wait(&send_reqs[i], &send_stats[i]);
                }
            }
        }

    }
    toc = MPI_Wtime();
    double l_elapsed_seconds = toc - tic, g_elapsed_seconds = 0;
    MPI_Reduce(&l_elapsed_seconds, &g_elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0)
        printf("time : %f secs\n", g_elapsed_seconds);


    /*if (myrank == 3) {
        GxB_print(Z[nlayers - 1], GxB_COMPLETE);
        GxB_print(W[nlayers - 1], GxB_COMPLETE);
        GxB_print(H[nlayers], GxB_COMPLETE);
    }*/

}

void read_buffer_sizes(char * path) {
    FILE * file = fopen(path, "r");

    send_buffer_sizes = (int64_t *) calloc(npes, sizeof (int64_t));
    recv_buffer_sizes = (int64_t *) calloc(npes, sizeof (int64_t));

    int ntargets, nsources;

    fscanf(file, "%d ", &ntargets);
    for (int i = 0; i < ntargets; i++) {
        int64_t target, size;
        fscanf(file, "%ld %ld", &target, &size);
        send_buffer_sizes[target] = size;

        if (max_buffer_size <= size)
            max_buffer_size = size;
    }

    Hrecv = (int *) malloc(npes * sizeof (int));
    for (int i = 0; i < npes; i++)
        Hrecv[i] = -1;

    fscanf(file, "%d ", &nsources);
    for (int i = 0; i < nsources; i++) {
        int64_t source, size;
        fscanf(file, "%ld %ld", &source, &size);
        recv_buffer_sizes[source] = size;

        if (max_buffer_size <= size)
            max_buffer_size = size;

        Hrecv[source] = 1;
    }

    /*if (myrank == 1) {
        printf("ntargets:%d\n", ntargets);
        for (int i = 0; i < npes; i++) {
            printf("%ld ", send_buffer_sizes[i]);
        }
        printf("\n");

        printf("nsources:%d\n", nsources);
        for (int i = 0; i < npes; i++) {
            printf("%ld ", recv_buffer_sizes[i]);
        }
        printf("\n");
    }*/

}

void statistics() {
    long long * stats_world, numstats = 20;
    long long total_comm_volume = 0, max_send_comm_volume = 0,
            max_recv_vol = 0, total_send_message_count = 0,
            max_send_message_count = 0, max_recv_msg = 0;

    MPI_Reduce(&stats[send_comm_volume], &total_comm_volume, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats[send_comm_volume], &max_send_comm_volume, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats[recv_comm_volume], &max_recv_vol, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats[send_message_count], &total_send_message_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats[send_message_count], &max_send_message_count, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&stats[recv_message_count], &max_recv_msg, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if (myrank == 0) {
        printf("%lld %lld %lld %lld %lld %lld %lld %lld\n", total_comm_volume, (total_comm_volume / npes), max_send_comm_volume, max_recv_vol,
                total_send_message_count, (total_send_message_count / npes), max_send_message_count, max_recv_msg);
    }

}

void read_connectivity(char * path) {
    FILE * f = fopen(path, "r");

    int ntargets;
    fscanf(f, "%d %d\n", &ntargets, &nrecvs);

    Hsend = (GrB_Matrix *) malloc(npes * sizeof (GrB_Matrix));
    for (int i = 0; i < npes; i++)
        Hsend[i] = NULL;

    for (int i = 0; i < ntargets; i++) {
        int target, nidx;
        fscanf(f, "%d %d\n", &target, &nidx);
        GrB_Matrix_new(&(Hsend[target]), GrB_FP32, nneurons[0], nneurons[0]);

        for (int j = 0; j < nidx; j++) {
            int idx;
            fscanf(f, "%d", &idx);
            GrB_Matrix_setElement(Hsend[target], 1, idx, idx);
        }

        GrB_Matrix_wait(&(Hsend[target]));
    }

    fclose(f);
}

void initialize_matrices() {
    srand(time(NULL));

    W = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    dW = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    //WT = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    H = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    //HT = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    AH = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));

    Hcap = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    Gcap = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    Z = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    G = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    AG = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));

    for (int layer = 0; layer < nlayers; layer++) {
        GrB_Matrix_new(&(H[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

    for (int layer = 0; layer < nlayers - 1; layer++) {
        GrB_Matrix_new(&(Hcap[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
        //GrB_Matrix_new(&(AH[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

    for (int layer = 1; layer < nlayers; layer++) {
        GrB_Matrix_new(&(W[layer]), GrB_FP32, nneurons[layer], nneurons[layer + 1]);
        //GrB_Matrix_new(&(dW[layer]), GrB_FP32, nneurons[layer], nneurons[layer + 1]);
        //GrB_Matrix_new(&(WT[layer]), GrB_FP32, nneurons[layer + 1], nneurons[layer]);
        //GrB_Matrix_new(&(Z[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);

        int ni = nneurons[layer], nout = nneurons[layer + 1];
        float sd = sqrt(6.0 / (float) (ni + nout));

        for (int i = 0; i < nneurons[layer]; i++) {
            for (int j = 0; j < nneurons[layer + 1]; j++) {
                float x = get_random(-sd, sd);
                GrB_Matrix_setElement(W[layer], x, i, j);
            }
        }

        GrB_Matrix_wait(&(W[layer]));
    }

    /*for (int layer = 0; layer < nlayers - 1; layer++) {
        GrB_Matrix_new(&(HT[layer]), GrB_FP32, nneurons[layer + 1], nneurons[0]);
    }*/

    for (int layer = 1; layer < nlayers; layer++) {
        GrB_Matrix_new(&(G[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
        //GrB_Matrix_new(&(AG[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
        GrB_Matrix_new(&(Gcap[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

}

void read_matrix(GrB_Matrix * M_ptr, char * path, int type) {
    FILE * file = fopen(path, "r");

    GrB_Index nvtx, nedges;
    fscanf(file, "%ld %ld\n", &nvtx, &nedges);

    GrB_Index *I = (GrB_Index *) malloc(nedges * sizeof (GrB_Index));
    GrB_Index *J = (GrB_Index *) malloc(nedges * sizeof (GrB_Index));
    float *X = malloc(nedges * sizeof (float));

    for (int k = 0; k < nedges; k++) {
        GrB_Index i, j;
        float x;
        fscanf(file, "%ld %ld %g\n", &i, &j, &x);
        I[k] = i;
        J[k] = j;
        X[k] = x;
    }

    GrB_Matrix C = NULL;
    (*M_ptr) = NULL;

    if (type == 0) {
        GrB_Matrix_new(&C, GrB_FP32, nneurons[0], nneurons[0]);
    } else if (type == 1) {
        GrB_Matrix_new(&C, GrB_FP32, nneurons[0], nneurons[1]);
    } else if (type == 2) {
        GrB_Matrix_new(&C, GrB_FP32, nneurons[0], nneurons[nlayers]);
    }


    GrB_Matrix_build(C, I, J, X, nedges, GrB_PLUS_FP32);
    (*M_ptr) = C;

    free(I);
    free(J);
    free(X);
    fclose(file);

}

void read_matrix2(GrB_Matrix * M_ptr, char * path) {
    FILE * file = fopen(path, "r");

    GrB_Index nrows;
    fscanf(file, "%ld\n", &nrows);

    GrB_Index *I = (GrB_Index *) malloc(nrows * nneurons[1] * sizeof (GrB_Index));
    GrB_Index *J = (GrB_Index *) malloc(nrows * nneurons[1] * sizeof (GrB_Index));
    float *X = malloc(nrows * nneurons[1] * sizeof (float));

    GrB_Index nvals = 0;
    for (GrB_Index k = 0; k < nrows; k++) {
        GrB_Index i, j;
        float x = 1.0;
        fscanf(file, "%ld\n", &i);
        for (j = 0; j < nneurons[1]; j++, nvals++) {
            I[nvals] = i;
            J[nvals] = j;
            X[nvals] = x;
        }
    }

    GrB_Matrix C = NULL;
    (*M_ptr) = NULL;

    GrB_Matrix_new(&C, GrB_FP32, nneurons[0], nneurons[1]);
    GrB_Matrix_build(C, I, J, X, nvals, GrB_PLUS_FP32);

    (*M_ptr) = C;

    free(I);
    free(J);
    free(X);
    fclose(file);

}

void read_config(char * path) {

    //FILE * file = fopen(path, "r");
    FILE * file = fopen(config_path, "r");

    fscanf(file, "%d", &nlayers);
    nneurons = (int *) malloc((nlayers + 1) * sizeof (int));
    for (int i = 0; i < nlayers + 1; i++) {
        int nneuron;
        fscanf(file, "%d ", &nneuron);
        nneurons[i] = nneuron;
    }

    if (myrank == 0) {
        printf("nlayers:%d\n", nlayers);
        for (int i = 0; i < nlayers + 1; i++)
            printf("%d ", nneurons[i]);
        printf("\n");
    }

    max_nfeatures = 0;
    for (int i = 1; i < nlayers + 1; i++) {
        if (max_nfeatures < nneurons[i]) {
            max_nfeatures = nneurons[i];
        }
    }

}

