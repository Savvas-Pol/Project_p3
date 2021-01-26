#include "help_functions.h"
#include "calculations.h"
#include "calculations_cluster.h"

using namespace std;

int main(int argc, char** argv) {
    string iFile, iFile2, classes, confFile, oFile, sline;
    int magic_number = 0, number_of_images = 0;
    int n_rows = 0, n_cols = 0, n_rows2 = 0, n_cols2 = 0, d, d2, count = 0;
    int k, L, kl, countl = 0, countw = 0;
    int changes = 6, first = 1;
    unsigned int dist;
    bool exists;
    char *cline, *pch;

    vector< vector<unsigned short> > pVec2, centroids2;
    vector<unsigned short> tempVec2, tempC2, pDim2;

    vector< vector<unsigned char> > pVec, centroids, centroids3, centroids_2_784;
    vector<unsigned char> tempVec, pDim, tempC;
    vector< vector<int> > clusters, clusters2, clusters3, temp, temp2, sVec;
    vector<int> aVec, tempIntVec, pos;
    vector< vector<distanceNode> > distRange;
    vector<distanceNode> distTemp;

    srand(time(NULL));

    read_inputCluster(&argc, argv, &iFile, &iFile2, &classes, &confFile, &oFile);
    read_confFile(&k, &L, &kl, confFile);

    // S2

    ofstream ofile(oFile);
    ifstream file(iFile);
    if (file.is_open()) {
        read_data(file, &magic_number, &number_of_images, &n_rows, &n_cols, pVec, tempVec);

		for(int i = 0; i < k; i++){
			clusters.push_back(vector<int>());
			temp.push_back(vector<int>());
		} 
        d = n_rows * n_cols;

		auto t3 = chrono::high_resolution_clock::now();

		k_means_init(centroids, number_of_images, pVec, k, d);

		if (ofile.is_open()){															//Lloyd's
			ofile << endl << "ORIGINAL SPACE" << endl;
			cout <<"clustering original space... " << endl;

			while((count < 40) && (changes > 5)){
				changes = 0;
				lloyds_assignment(clusters, temp, number_of_images, pVec, centroids, k, d, &changes, first);

				if(!first){
					if (changes <= 5)
						break;
				}
				else{
					changes = 6;
				}	

				centroids.erase(centroids.begin(), centroids.end());                      
				update_centroids_median(centroids, pDim, pVec, clusters, tempC, k, d);    		// new centroids

				first = 0;
				count++;
			}
			auto t4 = chrono::high_resolution_clock::now();
			auto durationLloyds = chrono::duration_cast<chrono::microseconds>( t4 - t3 ).count();

			for(int i=0; i<k; i++){
				ofile << "CLUSTER-" << i << " {size: " << clusters[i].size() << ", centroid: [";
				for (int y=0; y<d-1; y++){
					ofile << (int)centroids[i][y] << ", ";
				}
				ofile << (int)centroids[i][d-1] << "]}" << endl;
			}

			ofile << "clustering_time: " << durationLloyds << endl;
			silhouette(clusters, centroids, pVec, k, d, ofile);
			objective_function(centroids, pVec, k, d, ofile);
		}
		else{
			cout << "Output file does not exist." << endl;
		}
    }

    // S1

    temp.erase(temp.begin(), temp.end());
    first = 1;
    changes = 6;
    count = 0;
    
    ifstream file2(iFile2);
    if (file2.is_open()) {
        read_data2(file2, &magic_number, &number_of_images, &n_rows2, &n_cols2, pVec2, tempVec2);
        for (int i = 0; i < k; i++) {
            clusters2.push_back(vector<int>());
            temp2.push_back(vector<int>());
        }

        d2 = n_rows2 * n_cols2;

        auto t1 = chrono::high_resolution_clock::now();

        k_means_init2(centroids2, number_of_images, pVec2, k, d2);

        if (ofile.is_open()) { //Lloyd's
            ofile << "NEW SPACE" << endl;
            cout <<"clustering new space... " << endl;
            
            while ((count < 40) && (changes > 5)) {
                changes = 0;
                lloyds_assignment2(clusters2, temp2, number_of_images, pVec2, centroids2, k, d2, &changes, first);
                if (!first) {
                    if (changes <= 5)
                        break;
                } else {
                    changes = 6;
                }
                centroids2.erase(centroids2.begin(), centroids2.end());
                update_centroids_median2(centroids2, pDim2, pVec2, clusters2, tempC2, k, d2); // new centroids
                first = 0;
                count++;
            }
            auto t2 = chrono::high_resolution_clock::now();
            auto durationLloyds2 = chrono::duration_cast<chrono::microseconds>(t2 - t1).count();
            for (int i = 0; i < k; i++) {
                ofile << "CLUSTER-" << i << " {size: " << clusters2[i].size() << ", centroid: [";
                for (int y = 0; y < d2 - 1; y++) {
                    ofile << (int) centroids2[i][y] << ", ";
                }
                ofile << (int) centroids2[i][d2 - 1] << "]}" << endl;
            }
            ofile << "clustering_time: " << durationLloyds2 << endl;
            update_centroids_median(centroids_2_784, pDim, pVec, clusters2, tempC, k, d);
            
            silhouette(clusters2, centroids_2_784, pVec, k, d, ofile);
            objective_function(centroids_2_784, pVec, k, d, ofile);
        } else {
            cout << "Output file does not exist." << endl;
        }
    }
    
    //S3
    
    ifstream classesFile(classes);
    if (classesFile.is_open()) {
    	cout <<"clustering classification... " << endl;

        for (int i = 0; i < k; i++) {
            clusters3.push_back(vector<int>());
        }

        while (!classesFile.eof()) {
            getline(classesFile, sline);

            cline = new char[sline.length() + 1];
            strcpy(cline, sline.c_str());

            pch = strtok(cline, " ,()\n");
            while (pch != NULL) {
                countw++;

                if (countw > 3) {
                    clusters3[countl].push_back(atoi(pch));
                }
                pch = strtok(NULL, " ,()\n");
            }
            countw = 0;
            countl++;
        }
        update_centroids_median(centroids3, pDim, pVec, clusters3, tempC, k, d);
        if (ofile.is_open()) {
            ofile << "CLASSES AS CLUSTERS" << endl;
            silhouette(clusters3, centroids3, pVec, k, d, ofile);
            objective_function(centroids3, pVec, k, d, ofile);
        }
    }
    return 0;
}
