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

#include "patoh.h"
#include "omp.h"

using namespace std;
using namespace std::chrono;

typedef unordered_map<int, double> SpVector;
typedef unordered_map<int, SpVector> SpMatrix;

void read_matrix_market(string path, SpMatrix & A, int type);
void print_matrix(SpMatrix & A);
void partition_colnet();
void partition_random();
void print_parts(SpMatrix & A, int type);
void print_parts2(SpMatrix & A);
void print_connectivity();
void print_config();

int * partition_vector;
SpMatrix A, H, Y;
int k_way;
int nfeatures, nlayers, noutput_features = 2;
bool ishgp = true;
string path_to_A;
string path_to_H;
string path_to_Y;
string output_dir;

int c;

int main(int argc, char** argv) {

    while ((c = getopt(argc, argv, "a:h:o:k:f:l:y:r")) != -1) {
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
            case 'r':
                ishgp = false;
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
    printf("%d %d %d %d\n", k_way, nfeatures, nlayers, ishgp);

    read_matrix_market(path_to_A, A, 1);
    read_matrix_market(path_to_H, H, 1);
    read_matrix_market(path_to_Y, Y, 1);

    auto start = std::chrono::high_resolution_clock::now();
    if (ishgp) {
        partition_colnet();
    } else {
        partition_random();
    }
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;

    print_parts(A, 0);
    //print_parts(H, 1);
    print_parts2(H);
    print_parts(Y, 2);
    print_connectivity();
    print_config();

    printf("time: %f\n", elapsed.count());
    
    return 0;
}

void print_config() {
    ostringstream path;
    path << output_dir << "config";
    FILE * file = fopen(path.str().c_str(), "w+");

    fprintf(file, "%d %lu ", nlayers, A.size());
    for (int i = 0; i < nlayers - 1; i++) {
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

    printf("xx\n");
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

void partition_colnet() {
    PaToH_Parameters args;
    int _c, _n, _nconst, *cwghts, *nwghts,
            *xpins, *pins, cut, *partweights;

    _c = A.size();
    _n = A.size();
    _nconst = 1;

    nwghts = new int[_n];
    for (int i = 0; i < _n; i++) {
        nwghts[i] = 1;
    }

    cwghts = new int[_c * _nconst];
    for (int i = 0; i < _c; i++) {
        cwghts[i] = A[i].size();
    }

    xpins = new int[_n + 1];
    xpins[0] = 0;
    for (int i = 0; i < _n; i++) {
        xpins[i + 1] = A[i].size() + xpins[i];
    }

    pins = new int[xpins[_n]];
    for (int i = 0; i < _n; i++) {
        int j = xpins[i];
        for (auto e : A[i]) {
            pins[j++] = e.first;
        }
    }

    PaToH_Initialize_Parameters(&args, PATOH_CONPART, PATOH_SUGPARAM_DEFAULT);

    args._k = k_way;
    args.final_imbal = 0.001;
    partition_vector = new int[_c];
    partweights = new int[args._k * _nconst];

    /*args.MemMul_CellNet = 16;
    args.MemMul_Pins = 32;*/

    PaToH_Alloc(&args, _c, _n, _nconst, cwghts, nwghts,
            xpins, pins);

    PaToH_Part(&args, _c, _n, _nconst, 0, cwghts, nwghts,
            xpins, pins, NULL, partition_vector, partweights, &cut);

    printf("cut: %d\n", cut);

    int myconn = 0;
    unordered_set<int> conn;
    for (int i = 0; i < _n; i++) {
        conn.clear();
        for (int j = xpins[i]; j < xpins[i + 1]; j++) {
            conn.insert(partition_vector[pins[j]]);
        }

        myconn += conn.size() - 1;
    }
    printf("myconn:%d\n", myconn);


    free(cwghts);
    free(nwghts);
    free(xpins);
    free(pins);
    free(partweights);

    PaToH_Free();

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

