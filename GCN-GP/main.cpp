#include <stdio.h>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <unordered_map>
#include <regex> 
#include <unordered_set>
#include <bits/unordered_set.h>
#include <math.h>
#include <getopt.h>
#include <sys/stat.h>
#include <bits/unordered_map.h>
#include <chrono>
#include <random>

//#include "patoh.h"
#include "metis.h"
#include "omp.h"

using namespace std;
using namespace std::chrono;

typedef unordered_map<int, double> SpVector;
typedef unordered_map<int, SpVector> SpMatrix;

void read_matrix_market(string path, SpMatrix & A, int type);
void print_matrix(SpMatrix & A);
void partition_metis();
void partition_colnet_patoh();
void partition_random();
void print_parts(SpMatrix & A, int type);
void print_parts2(SpMatrix & A);
void print_connectivity();
void print_config();
void symmetrize();

int * partition_vector;
SpMatrix A, H, S, Y;
int k_way;
int nfeatures, nlayers, noutput_features = 2;
string path_to_A;
string path_to_H;
string path_to_Y;
string output_dir;

int c;

int main(int argc, char** argv) {

    while ((c = getopt(argc, argv, "a:h:o:k:f:l:y:")) != -1) {
        switch (c) {
            case 'a':
                path_to_A = string(optarg);
                break;
            case 'h':
                path_to_H = string(optarg);
                break;
            case 'y':
                path_to_Y = string(optarg);
                break;
            case 'o':
                output_dir = string(optarg);
                if (output_dir.back() != '/')
                    output_dir.push_back('/');
                break;
            case 'k':
                k_way = atoi(optarg);
                break;
            case 'f':
                nfeatures = atoi(optarg);
                break;
            case 'l':
                nlayers = atoi(optarg);
                break;
            case '?':
                fprintf(stderr, "Unknown option character `\\x%x'.\n", optopt);
            default:
                return 1;
                abort();
        }
    }

    printf("%s\n", path_to_A.c_str());
    printf("%s\n", path_to_H.c_str());
    printf("%s\n", path_to_Y.c_str());
    printf("%s\n", output_dir.c_str());
    printf("%d-way, %d features, %d layers\n", k_way, nfeatures, nlayers);

    read_matrix_market(path_to_A, A, 1);
    read_matrix_market(path_to_H, H, 1);
    read_matrix_market(path_to_Y, Y, 1);

    symmetrize();

    auto start = std::chrono::high_resolution_clock::now();
    partition_metis();
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    
    /*print_parts(A, 0);
    //print_parts(H, 1);
    print_parts2(H);
    print_parts(Y, 2);
    print_connectivity();
    print_config();*/

    printf("time: %f\n", elapsed.count());
    
    return 0;
}

void symmetrize() {
    for (int i = 0; i < A.size(); i++) {
        for (auto e : A[i]) {
            S[i][e.first] = 1;
            S[e.first][i] = 1;
        }
    }
}

void print_config() {
    ostringstream path;
    path << output_dir << "config";
    FILE * file = fopen(path.str().c_str(), "w+");

    fprintf(file, "%d %lu ", nlayers, A.size());
    for (int i = 0; i < nlayers-1; i++) {
        fprintf(file, "%d ", nfeatures);
    }
    fprintf(file, "%d ", noutput_features);
    fprintf(file, "\n");

    fclose(file);

}

void partition_random() {
    long seed = chrono::high_resolution_clock::now().time_since_epoch().count();
    mt19937 mt_rand(seed);
    uniform_int_distribution<int> dice_rand = uniform_int_distribution<int>(0, k_way - 1);

    partition_vector = new int[A.size()];

    for (int i = 0; i < A.size(); i++) {
        partition_vector[i] = dice_rand(mt_rand);
    }

}

void print_connectivity() {

    unordered_map<int, unordered_map<int, vector<int>>> Hsend;
    unordered_map<int, unordered_map<int, vector<int>>> Hrecv;

    int myconn = 0, source;
    unordered_set<int> targets;
    for (int i = 0; i < A.size(); i++) {
        source = partition_vector[i];
        targets.clear();

        for (auto e : A[i]) {
            int row = e.first;
            if (partition_vector[row] != source) {
                targets.insert(partition_vector[row]);
            }
        }

        if (targets.size() == 0)
            continue;

        myconn += targets.size();

        for (auto target : targets) {
            Hsend[source][target].push_back(i);
            Hrecv[target][source].push_back(i);
        }
    }

    printf("myconn:%d\n", myconn);

    for (int k = 0; k < k_way; k++) {
        ostringstream path;
        path << output_dir << "conn." << k;
        FILE * file = fopen(path.str().c_str(), "w+");
        fprintf(file, "%lu %lu\n", Hsend[k].size(), Hrecv[k].size());
        for (auto target : Hsend[k]) {
            fprintf(file, "%d %lu ", target.first, target.second.size());
            for (auto row : target.second) {
                fprintf(file, "%d ", row);
            }
            fprintf(file, "\n");
        }
        fclose(file);
    }

    for (int k = 0; k < k_way; k++) {
        ostringstream path;
        path << output_dir << "buff." << k;
        FILE * file = fopen(path.str().c_str(), "w+");
        
        fprintf(file, "%lu ", Hsend[k].size());
        for (auto & target : Hsend[k]) {
            fprintf(file, "%d %lu ", target.first, target.second.size());
        }
        
        fprintf(file, "\n%lu ", Hrecv[k].size());
        for (auto & source : Hrecv[k]) {
            fprintf(file, "%d %lu ", source.first, source.second.size());
        }
        
        fclose(file);
    }

}

void print_parts(SpMatrix & A, int type) {
    vector<unsigned long> part_weights(k_way, 0);
    for (int i = 0; i < A.size(); i++) {
        part_weights[partition_vector[i]] += A[i].size();
    }

    string name;
    switch (type) {
        case 0:
            name = "A.";
            break;
        case 1:
            name = "H.";
            break;
        case 2:
            name = "Y.";
    }

    for (int k = 0; k < k_way; k++) {
        ostringstream path;
        path << output_dir << name << k;
        FILE * file = fopen(path.str().c_str(), "w+");

        fprintf(file, "%lu %lu\n", A.size(), part_weights[k]);

        for (int i = 0; i < A.size(); i++) {
            if (partition_vector[i] != k)
                continue;
            for (auto e : A[i]) {
                fprintf(file, "%d %d %.2f\n", i, e.first, e.second);
            }
        }

        fclose(file);
    }

}

void print_parts2(SpMatrix & A) {
    vector<unsigned long> part_weights(k_way, 0);
    for (int i = 0; i < A.size(); i++) {
        part_weights[partition_vector[i]] += A[i].size();
    }

    string name = "H.";

    for (int k = 0; k < k_way; k++) {
        ostringstream path;
        path << output_dir << name << k;
        FILE * file = fopen(path.str().c_str(), "w+");

        int nrows = 0;
        for (int i = 0; i < A.size(); i++) {
            if (partition_vector[i] != k)
                continue;
            nrows++;
        }

        fprintf(file, "%d\n", nrows);

        for (int i = 0; i < A.size(); i++) {
            if (partition_vector[i] != k)
                continue;
            fprintf(file, "%d\n", i);
        }

        fclose(file);
    }

}


void partition_metis() {
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);

    options[METIS_OPTION_PTYPE] = METIS_PTYPE_KWAY;
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;  
    options[METIS_OPTION_UFACTOR] = 1;

    idx_t nvtxs = S.size();
    idx_t ncon = 1;
    idx_t nparts = k_way;

    idx_t nedges = 0;
    for (int i = 0; i < nvtxs; i++) {
        nedges += (S[i].size() - 1);
    }

    idx_t * xadj = (idx_t *) malloc((nvtxs + 1) * sizeof (idx_t));
    idx_t * adjncy = (idx_t *) malloc((nedges) * sizeof (idx_t));
    idx_t * vwgt = (idx_t *) malloc((nvtxs) * sizeof (idx_t));
    idx_t * adjwgt = (idx_t *) malloc((nedges) * sizeof (idx_t));

    idx_t * vsize = NULL;
    real_t * tpwgts = NULL;
    real_t * ubvec = NULL;

    idx_t edgecut;
    idx_t * part = (idx_t *) malloc((nvtxs) * sizeof (idx_t));

    xadj[0] = 0;
    for (int i = 0; i < nvtxs; i++) {
        xadj[i + 1] = xadj[i] + S[i].size() - 1;
        vwgt[i] = 1;
    }

    for (int i = 0, k = 0; i < nvtxs; i++) {
        for (auto e : S[i]) {
            if (e.first == i)
                continue;
            adjwgt[k] = 1;
            adjncy[k++] = e.first;
        }
    }

    idx_t res = METIS_PartGraphKway(&nvtxs, &ncon, xadj, adjncy, vwgt, vsize, adjwgt,
            &nparts, tpwgts, ubvec, options, &edgecut, part);

    if (res != METIS_OK) {
        printf("error code:%d\n", res);
    }

    printf("gp cut: %d\n", edgecut);

    partition_vector = new int[nvtxs];
    for (int i = 0; i < nvtxs; i++) {
        partition_vector[i] = (int) part[i];
    }

}

void print_matrix(SpMatrix & M) {
    for (int i = 0; i < M.size(); i++) {
        for (SpVector::iterator j = M[i].begin(); j != M[i].end(); j++) {
            printf("%d %d : %.1f\n", i + 1, j->first + 1, j->second);
        }
    }
}

void read_matrix_market(string path, SpMatrix & A, int type) {
    ifstream ifs(path);
    string line;

    getline(ifs, line);

    bool symmetric;
    if (line.find("symmetric") != string::npos) {
        symmetric = true;
    } else {
        symmetric = false;
    }

    while (getline(ifs, line)) {
        if (line.find("%") == string::npos)
            break;
    }

    istringstream iss(line);
    int nrow, ncol, nnz;
    iss >> nrow >> ncol >> nnz;

    for (int i = 0; i < nnz; i++) {
        getline(ifs, line);
        istringstream iss(line);
        int row, col;
        float w = 1;
        iss >> row >> col;
        if (type)
            iss >> w;

        row--;
        col--;

        A[row][col] = w;
        if (symmetric)
            A[col][row] = w;
    }

}

