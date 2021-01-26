#ifndef __CALCULATIONS_CLUSTER_H__
#define __CALCULATIONS_CLUSTER_H__

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
#include <vector>
#include <chrono>
#include <algorithm>
#include <climits>

#include "calculations.h"
#include "help_functions.h"

using namespace std;


void k_means_init(vector< vector<unsigned char> > &centroids, int number_of_images, vector< vector<unsigned char> > pVec, int k, int d);
void k_means_init2(vector< vector<unsigned short> > &centroids, int number_of_images, vector< vector<unsigned short> > pVec, int k, int d);
void lloyds_assignment(vector< vector<int> > &clusters, vector< vector<int> > temp, int number_of_images, vector< vector<unsigned char> > pVec, vector< vector<unsigned char> > centroids, int k, int d, int *changes, int first);
void lloyds_assignment2(vector< vector<int> > &clusters, vector< vector<int> > temp, int number_of_images, vector< vector<unsigned short> > pVec, vector< vector<unsigned short> > centroids, int k, int d, int *changes, int first);
void update_centroids_median(vector< vector<unsigned char> > &centroids, vector <unsigned char> pDim, vector< vector<unsigned char> > pVec, vector< vector<int> > clusters, vector <unsigned char> tempC, int k, int d);
void update_centroids_median2(vector< vector<unsigned short> > &centroids, vector <unsigned short> pDim, vector< vector<unsigned short> > pVec, vector< vector<int> > clusters, vector <unsigned short> tempC, int k, int d);

vector<distanceNode> approximate_range_search_clusterLSH(vector < vector<unsigned char> > centroids, vector < vector< vector <hTableNode> > > &lHashTables, int L, int pos, int d, double R, int cluster);

void silhouette(vector< vector<int> > clusters, vector< vector<unsigned char> > centroids, vector< vector<unsigned char> > pVec, int k, int d, ofstream &ofile);
void objective_function(vector< vector<unsigned char> > centroids, vector< vector<unsigned char> > pVec, int k, int d, ofstream &ofile);

#endif
