#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <math.h>
#include <malloc.h>
#include <getopt.h>

#include <stdio.h>
#include <string.h>

#include "mpi.h"
#include "GraphBLAS.h"


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

enum PARTITIONED_TIMES {
    data_comm_time, local_spmm_time, all_reduce_time, local_update_time
};
double partitioned_times[20];

int npes, myrank;
int nlayers;
int *nneurons;
float alpha = 0.01;

char path[256];
char config_path[256];
int nthreads = 1;
int max_nfeatures = 50;

GrB_Matrix A, Y;
GrB_Matrix * H, *AH, *AG, *W, *Z, *G, * dW, *Hcap;

void read_matrix(GrB_Matrix * M_ptr, char * path, int type);
void read_matrix2(GrB_Matrix * M_ptr, char * path);
void GCN();
void initialize_matrices();
void read_config(char * path);
void statistics();

int c;

int main(int argc, char *argv[]) {

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

    mallopt(M_MMAP_MAX, 0);
    mallopt(M_TRIM_THRESHOLD, -1);
    mallopt(M_TOP_PAD, 16 * 1024 * 1024);
    GxB_init(GrB_NONBLOCKING, malloc, calloc, realloc, free, true);
    GxB_set(GxB_NTHREADS, nthreads);

    char path_A[512];
    char path_H[512];
    char path_Y[512];
    sprintf(path_A, "%sA.%d", path, myrank);
    sprintf(path_H, "%sH.%d", path, myrank);
    sprintf(path_Y, "%sY.%d", path, myrank);

    read_config(config_path);
    initialize_matrices();

    read_matrix(&A, path_A, 0);
    read_matrix2(&H[0], path_H);
    read_matrix(&Y, path_Y, 2);

    GCN();

    statistics();

    GrB_finalize();
    MPI_Finalize();

    return 0;
}

void sigmoid(void *z, const void *x) {
    (*(float *) z) = 1 / (1 + expf(-*(float *) x));
    //(*(float *) z) = *(float *) x;
}

void GCN() {

    GrB_Index local_nrows, nrows;
    GrB_Matrix_nrows(&nrows, H[0]);
    local_nrows = nrows / npes * 2;

    GrB_Index *I = (GrB_Index *) malloc(max_nfeatures * local_nrows * sizeof (GrB_Index));
    GrB_Index *J = (GrB_Index *) malloc(max_nfeatures * local_nrows * sizeof (GrB_Index));
    float * X = (float *) malloc(max_nfeatures * local_nrows * sizeof (float));

    int comm_buffer_size = (3 * max_nfeatures * local_nrows + 1) * sizeof (GrB_Index);
    char * comm_buffer = (char *) malloc(comm_buffer_size);

    GrB_UnaryOp sigmoid_op = NULL;
    GrB_UnaryOp_new(&sigmoid_op, sigmoid, GrB_FP32, GrB_FP32);

    GrB_Index nvals;

    stats[recv_comm_volume] = 0;
    stats[send_comm_volume] = 0;
    stats[recv_message_count] = 0;
    stats[send_message_count] = 0;

    partitioned_times[data_comm_time] = 0;
    partitioned_times[local_spmm_time] = 0;
    partitioned_times[all_reduce_time] = 0;
    partitioned_times[local_update_time] = 0;



    double t1, t2, dt;
    double tic, toc;
    tic = MPI_Wtime();
    for (int epoch = 0; epoch < 5; epoch++) {
        for (int layer = 1; layer < nlayers; layer++) {
            OK(GrB_Matrix_new(&(AH[layer - 1]), GrB_FP32, nneurons[0], nneurons[layer]));
            for (int i = 0; i < npes; i++) {
                if (myrank == i) {
                    OK(GrB_Matrix_nvals(&nvals, H[layer - 1]));
                    OK(GrB_Matrix_extractTuples(I, J, X, &nvals, H[layer - 1]));
                    int position = 0;
                    MPI_Pack(&nvals, 1, MPI_UNSIGNED_LONG, comm_buffer, comm_buffer_size, &position, MPI_COMM_WORLD);
                    MPI_Pack(I, nvals, MPI_UNSIGNED_LONG, comm_buffer, comm_buffer_size, &position, MPI_COMM_WORLD);
                    MPI_Pack(J, nvals, MPI_UNSIGNED_LONG, comm_buffer, comm_buffer_size, &position, MPI_COMM_WORLD);
                    MPI_Pack(X, nvals, MPI_FLOAT, comm_buffer, comm_buffer_size, &position, MPI_COMM_WORLD);

                    t1 = MPI_Wtime();
                    MPI_Bcast(comm_buffer, comm_buffer_size, MPI_PACKED, i, MPI_COMM_WORLD);
                    t2 = MPI_Wtime();
                    dt = t2 - t1;
                    partitioned_times[data_comm_time] += dt;

                    stats[send_comm_volume] += nvals;
                    stats[send_message_count]++;

                    OK(GrB_mxm(AH[layer - 1], NULL, GrB_PLUS_FP32, GxB_PLUS_TIMES_FP32,
                            A, H[layer - 1], GrB_NULL));
                } else {

                    t1 = MPI_Wtime();
                    MPI_Bcast(comm_buffer, comm_buffer_size, MPI_PACKED, i, MPI_COMM_WORLD);
                    t2 = MPI_Wtime();
                    dt = t2 - t1;
                    partitioned_times[data_comm_time] += dt;

                    int position = 0;
                    MPI_Unpack(comm_buffer, comm_buffer_size, &position, &nvals, 1, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                    MPI_Unpack(comm_buffer, comm_buffer_size, &position, I, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                    MPI_Unpack(comm_buffer, comm_buffer_size, &position, J, nvals, MPI_UNSIGNED_LONG, MPI_COMM_WORLD);
                    MPI_Unpack(comm_buffer, comm_buffer_size, &position, X, nvals, MPI_FLOAT, MPI_COMM_WORLD);
                    OK(GrB_Matrix_clear(Hcap[layer - 1]));
                    OK(GrB_Matrix_build(Hcap[layer - 1], I, J, X, nvals, GrB_PLUS_FP32));
                    OK(GrB_mxm(AH[layer - 1], NULL, GrB_PLUS_FP32, GxB_PLUS_TIMES_FP32,
                            A, Hcap[layer - 1], GrB_NULL));

                    stats[recv_comm_volume] += nvals;
                    stats[recv_message_count]++;
                }
            }
            OK(GrB_Matrix_new(&(Z[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]));
            OK(GrB_mxm(Z[layer], NULL, NULL, GxB_PLUS_TIMES_FP32, AH[layer - 1], W[layer], GrB_NULL));
            OK(GrB_Matrix_free(&(AH[layer - 1])));
            OK(GrB_apply(H[layer], NULL, NULL, sigmoid_op, Z[layer], GrB_DESC_R));
        }
    }
    toc = MPI_Wtime();
    double l_elapsed_seconds = toc - tic, g_elapsed_seconds = 0;
    MPI_Reduce(&l_elapsed_seconds, &g_elapsed_seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (myrank == 0)
        printf("time : %f secs\n", g_elapsed_seconds);

    /*if (myrank == 3) {
        GxB_print(H[nlayers - 1], GxB_COMPLETE);
    }*/

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

void read_config(char * path) {

    FILE * file = fopen(path, "r");

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

float get_random(float x, float y) {
    float r = (float) rand() / (float) RAND_MAX;
    return x + (y - x) * r;
}

void initialize_matrices() {
    srand(time(NULL));

    W = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    dW = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    H = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    AH = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    Hcap = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));

    Z = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    G = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));
    AG = (GrB_Matrix *) malloc((nlayers) * sizeof (GrB_Matrix));

    for (int layer = 0; layer < nlayers; layer++) {
        GrB_Matrix_new(&(H[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

    for (int layer = 0; layer < nlayers - 1; layer++) {
        GrB_Matrix_new(&(Hcap[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

    for (int layer = 1; layer < nlayers; layer++) {
        GrB_Matrix_new(&(W[layer]), GrB_FP32, nneurons[layer], nneurons[layer + 1]);

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

    for (int layer = 1; layer < nlayers; layer++) {
        GrB_Matrix_new(&(G[layer]), GrB_FP32, nneurons[0], nneurons[layer + 1]);
    }

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


    double avg_data_comm_time, avg_local_spmm_time, avg_all_reduce_time, avg_local_update_time;

    MPI_Reduce(&partitioned_times[data_comm_time], &avg_data_comm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&partitioned_times[local_spmm_time], &avg_local_spmm_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&partitioned_times[all_reduce_time], &avg_all_reduce_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&partitioned_times[local_update_time], &avg_local_update_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (myrank == 0) {



        avg_data_comm_time /= npes;
        avg_local_spmm_time /= npes;
        avg_all_reduce_time /= npes;
        avg_local_update_time /= npes;

        avg_local_spmm_time += avg_local_update_time;

        printf("%.6f %.6f %.6f\n", avg_data_comm_time, avg_local_spmm_time, avg_all_reduce_time);
    }

}
