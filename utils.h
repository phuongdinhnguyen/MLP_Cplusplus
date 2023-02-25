#pragma once
#include <vector>
#include "matrix.h"

using namespace std;

int convertToLabel(vector<double>data)
{
    int pos = 0;
    double max = data[0];
    for (int i = 1; i < data.size(); i++)
    {
        if (data[i] > max)
        {
            max = data[i];
            pos = i;
        }
    }
    return pos;
}

Matrix OneHotCoding(vector<int> label, int numClass)
{
    int numLabel = label.size();
    Matrix res(numClass, numLabel);

    fill(res.data.begin(), res.data.end(), 0);
// #pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < numLabel; i++)
    {
        int pos = label[i];
        res.data[pos * numLabel + i] = 1;
    }

    return res;
}

void CheckAccuracy(Matrix outputLayer, vector <int> trainLabel, int numData)
{
    // MNIST output layer should be: 10 x numData
    double totalAcc = 0;
    for (int colIdx = 0; colIdx < outputLayer.col; colIdx++)
    {
        vector <double> data;
        for (int rowIdx = 0; rowIdx < outputLayer.row; rowIdx++)
        {
            data.push_back(outputLayer.at(rowIdx, colIdx));
        }
        totalAcc += (trainLabel[colIdx] == convertToLabel(data));
    }
    cout << "accuracy: " << totalAcc / numData << endl;
}