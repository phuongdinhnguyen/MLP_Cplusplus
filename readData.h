#pragma once
#include "matrix.h"
#include <fstream>

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
    ifstream file("./mnist/train-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        cout << "reading train pictures! \n";
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
        cout << "reading train labels! \n";
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