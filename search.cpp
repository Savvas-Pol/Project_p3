#include "help_functions.h"
#include "calculations.h"
#include "calculations_lsh.h"

#define w 40000
#define N 1
// #define m 107					//a_max < m < M/2
#define NForTable 16

using namespace std;

int main(int argc, char** argv) {
    string iFile, iFile2, qFile, qFile2, oFile;
    int i, k, L;
    int magic_number = 0, number_of_images = 0;
    int n_rows = 0, n_cols = 0, n_rows2 = 0, n_cols2 = 0;
    int d, d2, M, m, h, pos;
    unsigned int g, rDist;
    int hTableSize, probes;
    long long lshSum = 0, trueSum = 0, redSum = 0;
    double approximationLsh, approximationReduced;

    vector< vector<unsigned char> > pVec, qVec;
    vector<unsigned char> tempVec;
    vector< vector<unsigned short> > pVec2, qVec2;
    vector<unsigned short> tempVec2;
    vector< vector<int> > sVec;
    vector<int> aVec, tempIntVec;
    vector<distanceNode> distLsh, distTrue, distTrue2;

    read_inputLSH(&argc, argv, &iFile, &iFile2, &qFile, &qFile2, &k, &L, &oFile);

    M = pow(2, floor(32 / k));
    m = (M / 2) - 1;

    ifstream file(iFile);
    ifstream file2(iFile2);
    if (file.is_open() && file2.is_open()) {
        read_data(file, &magic_number, &number_of_images, &n_rows, &n_cols, pVec, tempVec);
        d = n_rows * n_cols; // dimension

        read_data2(file2, &magic_number, &number_of_images, &n_rows2, &n_cols2, pVec2, tempVec2);
        d2 = n_rows2 * n_cols2; // dimension

        hTableSize = number_of_images / NForTable;

        vector < vector< vector <hTableNode> > > lHashTables; // vector with L hash tables
        vector< vector <hTableNode> > hashTable; // hash table

        create_hashtables_LSH(lHashTables, hashTable, pVec, L, hTableSize, k, d, number_of_images, w, m, M);

        ifstream qfile(qFile);
        ifstream qfile2(qFile2);
        if (qfile.is_open() && qfile2.is_open()) {
            read_data(qfile, &magic_number, &number_of_images, &n_rows, &n_cols, qVec, tempVec);
            read_data2(qfile2, &magic_number, &number_of_images, &n_rows2, &n_cols2, qVec2, tempVec2);

            ofstream ofile(oFile);
            if (ofile.is_open()) {
                for (int i = 0; i < k; i++) {
                    tempIntVec = get_s(w, d); //s_i uniform random generator
                    sVec.push_back(tempIntVec);
                    tempIntVec.erase(tempIntVec.begin(), tempIntVec.end());
                }

                int countLsh = 0;
                for (int i = 0; i < number_of_images; i++) {
                    for (int j = 0; j < k; j++) {
                        aVec = calculate_a(qVec[i], sVec[j], w, d); // calculate a for every image
                        h = calculate_h(aVec, m, M, d); // calculate h for every image
                        tempIntVec.push_back(h);
                    }
                    g = calculate_g(tempIntVec, k); // calculate g for every image
                    pos = g % hTableSize; // find the position to insert the image in the hash table
                    tempIntVec.erase(tempIntVec.begin(), tempIntVec.end());

                    auto t5 = chrono::high_resolution_clock::now();
                    distTrue2 = actual_nearest_neighbor2(qVec2[i], pVec2, d2, N);
                    auto t6 = chrono::high_resolution_clock::now();
                    auto durationTrue2 = chrono::duration_cast<chrono::microseconds>(t6 - t5).count();

                    auto t1 = chrono::high_resolution_clock::now();
                    distLsh = approximate_nearest_neighbor(qVec[i], lHashTables, L, pos, d, N, g);
                    auto t2 = chrono::high_resolution_clock::now();
                    auto durationLsh = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();

                    auto t3 = chrono::high_resolution_clock::now();
                    distTrue = actual_nearest_neighbor(qVec[i], pVec, d, N);
                    auto t4 = chrono::high_resolution_clock::now();
                    auto durationTrue = chrono::duration_cast<chrono::microseconds>(t4 - t3).count();

                    ofile << "Query: " << i << endl; // write to file
                    for (int j = 0; j < N; j++) {
                        rDist = manhattan_dist(qVec[i], pVec[distTrue2[j].pPos], d);
                        ofile << "Nearest neighbor Reduced: " << distTrue2[j].pPos << endl;
                        ofile << "Nearest neighbor LSH: " << distLsh[j].pPos << endl;
                        ofile << "Nearest neighbor True: " << distTrue[j].pPos << endl;
                        ofile << "distanceReduced: " << rDist << endl;
                        ofile << "distanceLSH: " << distLsh[j].dist << endl;
                        ofile << "distanceTrue: " << distTrue[j].dist << endl;

                        if (distLsh[j].pPos != -1) {
                            lshSum += distLsh[j].dist;
                            countLsh++;
                        }
                        trueSum += distTrue[j].dist;
                        redSum += rDist;
                    }
                    ofile << "tReduced: " << durationTrue2 << endl;
                    ofile << "tLSH: " << durationLsh << endl;
                    ofile << "tTrue: " << durationTrue << endl;
                }
                cout << countLsh << endl;
                approximationLsh = (double) (lshSum / countLsh) / (double) (trueSum / number_of_images);
                approximationReduced = (double) (redSum / number_of_images) / (double) (trueSum / number_of_images);

                ofile << "Approximation Factor LSH: " << approximationLsh << endl;
                ofile << "Approximation Factor Reduced: " << approximationReduced << endl;
            }
        } else {
            cout << "Could not open query file." << endl;
            return 0;
        }
    } else {
        cout << "Could not open input file." << endl;
        return 0;
    }
    return 0;
}
