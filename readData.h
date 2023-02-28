#pragma once
#include "matrix.h"
#include <fstream>
#include <string>

using namespace std;

/*------ Read data from MNIST file ----------*/
int reverseInt(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}
/*-------------------------------------------*/

void ReadDataMNIST(Matrix& inputLayer, vector <int>& trainLabel, int numData)
{
    // Read picture pixel
    inputLayer.data.resize(0);
    inputLayer.row = numData;
    inputLayer.col = 784;
    ifstream file("./mnist/train-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        cout << "reading train pictures... \n";
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file.read((char*)&number_of_images, sizeof(number_of_images));
        number_of_images = reverseInt(number_of_images);
        file.read((char*)&n_rows, sizeof(n_rows));
        n_rows = reverseInt(n_rows);
        file.read((char*)&n_cols, sizeof(n_cols));
        n_cols = reverseInt(n_cols);
        for (int i = 0; i < numData; ++i)
        {
            for (int j = 0; j < n_rows * n_cols; ++j)
            {
                unsigned char temp = 0;
                file.read((char*)&temp, sizeof(temp));
                inputLayer.data.push_back(temp / 255.0);
                // if (j % 28 == 0) cout << endl;
                // cout << (int)temp << " ";
            }
        }
    }
    else
    {
        cout << "Could not open data file! \n";
    }

    file.close();
    // matrix::showMatrix(inputLayer);

    // Read train label
    ifstream file2("./mnist/train-labels.idx1-ubyte", ios::binary);
    if (file2.is_open())
    {
        cout << "reading train labels... \n";
        int magic_number = 0;
        int number_of_items = 0;

        file2.read((char*)&magic_number, sizeof(magic_number));
        magic_number = reverseInt(magic_number);
        file2.read((char*)&number_of_items, sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);

        for (int i = 0; i < numData; ++i)
        {
            unsigned char temp = 0;
            file2.read((char*)&temp, sizeof(temp));
            trainLabel.push_back((int)temp);
        }
    }
    else
    {
        cout << "Could not open label file! \n";
    }
    file2.close();
}

string CIFAR10_file[5] = { "./cifar-10/data_batch_1.bin","./cifar-10/data_batch_2.bin","./cifar-10/data_batch_3.bin",
                          "./cifar-10/data_batch_4.bin","./cifar-10/data_batch_5.bin" };


void ReadDataCIFAR10(Matrix& inputLayer, vector <int>& trainLabel, int numFiles, int numDataEachFile)
{
    int numberOfData = numDataEachFile;
    int numberOfFile = numFiles;
    int dim = 3072;
    ifstream file3;

    inputLayer.data.resize(0);
    inputLayer.row = numFiles * numDataEachFile;
    inputLayer.col = dim;

    vector <double> data;
    vector <int> labels;
    for (int file_idx = 0; file_idx < numberOfFile; file_idx++)
    {
        file3.open(CIFAR10_file[file_idx], ios::binary);
        if (file3.is_open())
        {
            cout << "file opened! \n";
            unsigned char label = 0;
            for (int cnt = file_idx * numberOfData; cnt < (file_idx + 1) * numberOfData; cnt++)
            {
                file3.read((char*)&label, sizeof(label));
                labels.push_back((int)label);
                for (int i = 0; i < dim; ++i)
                {
                    unsigned char pixel = 0;
                    file3.read((char*)&pixel, sizeof(pixel));
                    //m_dataPoints[cnt * dim + i] = (int)pixel;
                    data.push_back((double)pixel / 255.0);
                }
            }
        }
        file3.close();
    }
    std::copy(data.begin(), data.end(), inputLayer.data.begin());
}

string CIFAR10_test = "./cifar-10/test_batch.bin";
void ReadDataCIFAR10Test(Matrix& testData, vector <int>& testLabel)
{
    int numberOfData = 10000;
    int dim = 3072;
    ifstream file3;

    testData.data.resize(0);
    testData.row = numberOfData;
    testData.col = dim;

    vector <double> data;
    vector <int> labels;
    file3.open(CIFAR10_test, ios::binary);
    if (file3.is_open())
    {
        cout << "open and reading test data... \n";
        unsigned char label = 0;
        for (int cnt = 0; cnt < numberOfData; cnt++)
        {
            file3.read((char*)&label, sizeof(label));
            labels.push_back((int)label);
            for (int i = 0; i < dim; ++i)
            {
                unsigned char pixel = 0;
                file3.read((char*)&pixel, sizeof(pixel));
                data.push_back((double)pixel / 255.0);
            }
        }
    }
    file3.close();
    std::copy(data.begin(), data.end(), testData.data.begin());
}