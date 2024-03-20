#ifndef HOTS_H
#define HOTS_H

#include <vector>
#include <array>
#include "common.hpp"

//using namespace cv;
using namespace std;

// holds center data for the model
// 3 points (x, y, time surface)
// variable length(vector) number of centers per layer
// each layer has centers
vector< vector< vector<float>>> modelCenters {};

// holds histogram data for classification
// bucket for each feature in last layer
// histogram for each class
array< array<unsigned int, NF * KN * (LAYERS - 1)>, CLASSES> modelHistograms {};

// vector of histogram (ordered) labels
vector<string> labels {};

// sort array of arrays by timestamp
bool timeSort(const array<unsigned int, 4>, const array<unsigned int, 4>);

// read model file
int readModel();

// ANN for MLP classifier
//Ptr<cv::ml::ANN_MLP> ann;

#endif /* HOTS_H */