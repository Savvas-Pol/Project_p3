#ifndef __HELP_FUNCTIONS_H__
#define __HELP_FUNCTIONS_H__

#include <iostream>
#include <cstdlib>
#include <fstream>
#include <bits/stdc++.h>
#include <string>
#include <list> 
#include <iterator> 
#include <random>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>


using namespace std;

int reverseInt (int i);
void read_data(ifstream &file, int* magic_number, int* number_of_images, int* n_rows, int* n_cols, vector< vector<unsigned char> >& pVec, vector<unsigned char>& tempVec);
void read_data2(ifstream &file, int* magic_number, int* number_of_images, int* n_rows, int* n_cols, vector< vector<unsigned short> >& pVec, vector<unsigned short>& tempVec);

void read_inputLSH(int* argc, char** argv, string* iFile, string* iFile2, string* qFile, string* qFile2, int* k, int* L, string* oFile);
void read_inputCluster(int* argc, char** argv, string* iFile, string* iFile2, string* classes, string* confFile, string* oFile);

void read_confFile(int* K, int* L, int* kl, string confFile);

void quicksort(vector<unsigned char> &values, int left, int right);
int partition(vector<unsigned char> &values, int left, int right);

void quicksort2(vector<unsigned short> &values, int left, int right);
int partition2(vector<unsigned short> &values, int left, int right);

#endif
