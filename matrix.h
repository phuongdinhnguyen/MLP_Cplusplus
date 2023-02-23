#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>

#include <time.h>
#include <chrono>
#include <ctime>


using namespace std;

#define NUM_THREAD 6

//  Define 2D Matrix
struct Matrix {
    int row;
    int col;
    vector <float> data;
    Matrix() {}
    Matrix(int x, int y) { row = x, col = y; data.resize(row * col); }
    float at(int x, int y)
    {
        // extract data at location (x,y)
        return data[x * col + y];
    }
};

namespace matrix
{

    void CheckErr(int x, int y, string errThrow)
    {
        try
        {
            if (x != y) throw;
        }
        catch (const std::exception& e)
        {
            std::cout << errThrow << std::endl;
        }
    }

    void showMatrix(Matrix x)
    {
        for (int i = 0; i < x.row; i++)
        {
            cout << "Row " << i << ":\t";
            for (int j = 0; j < x.col; j++)
            {
                cout << x.at(i, j) << " ";
            }
            cout << endl;
        }
    }

    void matrixSize(Matrix x)
    {
        cout << "Matrix: " << x.row << " " << x.col << endl;
    }

    Matrix transpose(Matrix x)
    {
        Matrix res(x.col, x.row);
        res.data.resize(x.col * x.row);
        #pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < x.row; i++)
        {
            for (int j = 0; j < x.col; j++)
                res.data[j * res.col + i] = x.at(i, j);
        }
        return res;
    }

    Matrix matrix_dot(Matrix x1, Matrix x2)
    {
        // dot operator for matrix: (x1.row, x1.col) . (x2.row, x2.col)

        // Checking matrix errors
        CheckErr(x1.col, x2.row, "Col of matrix 1 and Row of matrix 2 invalid!");

        //  Do the multiplication
        
        Matrix x2_t = matrix::transpose(x2);
        

        Matrix res(x1.row, x2.col);

        auto startTime = std::chrono::steady_clock::now();
        #pragma omp parallel
        {
            int i, j, k;
            #pragma omp for
            for (i = 0; i < res.row; i++) { 
                for (j = 0; j < res.col; j++) {
                    float dot  = 0;
                    for (k = 0; k < x2_t.col; k++) {
                        dot += x1.at(i,k)*x2_t.at(j,k);
                    } 
                    res.data[i * (res.col) + j] = dot;
                }
            }
        }
        auto endTime = std::chrono::steady_clock::now();
        auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        cout << "Total time for this function: " << encTime / 1000.0 << " sec." << endl;

        return res;
    }

    Matrix matrix_plus(Matrix x1, Matrix x2)
    {
        // plus operator for matrix: (x1.row, x1.col) + (x2.row, x2.col)

        // Checking matrix errors
        CheckErr(x1.row, x2.row, "Sum of 2 matrix not same row!");
        CheckErr(x1.col, x2.col, "Sum of 2 matrix not same col!");

        // Do the plus
        Matrix res(x1.row, x1.col);
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.row; i++)
            for (int j = 0; j < res.col; j++)
                res.data[i * res.col + j] = x1.at(i, j) + x2.at(i, j);

        return res;
    }

    Matrix matrix_minus(Matrix x1, Matrix x2)
    {
        // subtract(minus) operator for matrix: (x1.row, x1.col) - (x2.row, x2.col)

        // Checking matrix errors
        CheckErr(x1.row, x2.row, "Subtract of 2 matrix not same row!");
        CheckErr(x1.col, x2.col, "Subtract of 2 matrix not same col!");

        // Do the plus
        Matrix res(x1.row, x1.col);
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.row; i++)
            for (int j = 0; j < res.col; j++)
                res.data[i * res.col + j] = x1.at(i, j) - x2.at(i, j);

        return res;
    }

    // Divide all matrix with a number
    Matrix divAll(Matrix x, float num)
    {
        Matrix res = x;

#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = res.data[i] / num;
        }
        return res;
    }

    // Multiply all matrix with a number
    Matrix mulAll(Matrix x, float num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = res.data[i] * num;
        }
        return res;
    }

    // Minus all matrix with a number
    Matrix minusAll(Matrix x, float num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = res.data[i] - num;
        }
        return res;
    }

    // Multiply all elements of 2 matrix together
    Matrix mulAllMatrix(Matrix x, Matrix a)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = x.data[i] * a.data[i];
        }
        return res;
    }

    // Sum each row of matrix
    Matrix sumRow(Matrix x)
    {
        Matrix res;
        res.row = x.row;
        res.col = 1;
        res.data.resize(res.row * res.col);
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
        {
            float sum = 0;
            for (int colIdx = 0; colIdx < x.col; colIdx++)
            {
                sum += x.at(rowIdx, colIdx);
            }
            res.data[rowIdx] = sum;
        }

        return res;
    }

    // Sum all elements in matrix
    float sumAllMatrix(Matrix x)
    {
        float sum = 0;
        for (int i = 0; i < x.col * x.row; i++)
            sum += x.data[i];

        return sum;
    }

 
}