#pragma once
#include <iostream>
#include <vector>
#include <string>
#include <omp.h>
#include <random>

#include <time.h>
#include <chrono>
#include <ctime>


using namespace std;

#define NUM_THREAD 6
#define DEBUG 0
//  Define 2D Matrix
struct Matrix {
    int row;
    int col;
    vector <double> data;
    Matrix() {}
    Matrix(int x, int y) { row = x, col = y; data.resize(row * col); }
    double at(int x, int y)
    {
        // extract data at location (x,y)
        return data[x * col + y];
    }

    void initSize(int x, int y)
    {
        row = x, col = y; data.resize(row * col);
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

    void RandMatrix(Matrix& x)
    {
        int maxLength = 10000;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        static std::default_random_engine generator(90);
        std::uniform_int_distribution<int> distribution(0, maxLength);

        for (int i = 0; i < x.col * x.row; i++)
        {
            x.data[i] = (distribution(generator) * 1.0 / maxLength - 0.5) / 100;
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

    void showMatrix(Matrix x, int col)
    {
        for (int i = 0; i < x.row; i++)
        {
            cout << "Row " << i << ":\t";
            for (int j = 0; j < col; j++)
            {
                cout << x.at(i, j) << " ";
            }
            cout << endl;
        }
    }

    void checkMatrix(Matrix x)
    {
        bool flag = 1;
        for (int i = 0; i < x.row * x.col; i++)
        {
            if (isnan(x.data[i]) || isinf(x.data[i]))
            {
                cout << "NaN / INF found!\n";
                return;
            }
        }
        cout << "good matrix!\n";
    }

    void matrixSize(Matrix x)
    {
        cout << "Matrix: " << x.row << " " << x.col << endl;
    }

    Matrix transpose(Matrix x)
    {
#if DEBUG
        auto startTime = std::chrono::steady_clock::now();
#endif
        Matrix res(x.col, x.row);
        res.data.resize(x.col * x.row);
        int i, j;
        #pragma omp parallel for private(j) num_threads(NUM_THREAD)
        for (i = 0; i < x.row; i++)
        {
            for (j = 0; j < x.col; j++)
            {
                res.data[j * res.col + i] = isnan(x.at(i, j)) ? 0 : x.at(i, j);
                //cout << i << " " << j << endl;
            }
        }
#if DEBUG
        auto endTime = std::chrono::steady_clock::now();
        auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        cout << "Total time for transpose: " << encTime / 1000.0 << " sec." << endl;
#endif
        return res;
    }

    Matrix matrix_dot(Matrix& x1, Matrix& x2)
    {
        // dot operator for matrix: (x1.row, x1.col) . (x2.row, x2.col)

        // Checking matrix errors
        CheckErr(x1.col, x2.row, "Col of matrix 1 and Row of matrix 2 invalid!");

        //  Do the multiplication
        
        Matrix x2_t = matrix::transpose(x2);
        
        
        Matrix res(x1.row, x2.col);
#if DEBUG
        auto startTime = std::chrono::steady_clock::now();
#endif
        #pragma omp parallel
        {
            int i, j, k;
            #pragma omp for
            for (i = 0; i < res.row; i++) { 
                for (j = 0; j < res.col; j++) {
                    double dot  = 0;
                    for (k = 0; k < x2_t.col; k++) {
                        double tmp = x1.at(i, k) * x2_t.at(j, k);
                        dot += isnan(tmp) ? 0 : tmp;
                    } 
                    res.data[i * (res.col) + j] = isnan(dot) ? 0 : dot;
                }
            }
        }
#if DEBUG
        auto endTime = std::chrono::steady_clock::now();
        auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        cout << "Total time for dot function: " << encTime / 1000.0 << " sec." << endl;
#endif
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
        Matrix res= x1;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.row; i++)
            for (int j = 0; j < res.col; j++)
            {
                res.data[i * res.col + j] = x1.at(i, j) - x2.at(i, j);
                res.data[i * res.col + j] = isnan(res.data[i * res.col + j]) ? 0 : res.data[i * res.col + j];
            }
                

        return res;
    }

    // Divide all matrix with a number
    Matrix divAll(Matrix x, double num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = (isnan(res.data[i]) ? 0 : res.data[i]) / num;
        }
        return res;
    }

    // Multiply all matrix with a number
    Matrix mulAll(Matrix x, double num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            double tmp = res.data[i] * num;
            res.data[i] = isnan(tmp) ? 0 : tmp;
        }
        return res;
    }

    // Minus all matrix with a number
    Matrix minusAll(Matrix x, double num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            double tmp = res.data[i] - num;
            res.data[i] = isnan(tmp) ? 0 : tmp;
        }
        return res;
    }

    // plus all matrix with a number
    Matrix plusAll(Matrix x, double num)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = res.data[i] + num;
            //if (isnan(res.data[i])) res.data[i] = 0;
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
            //if (isnan(res.data[i])) res.data[i] = 0;
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
            double sum = 0;
            for (int colIdx = 0; colIdx < x.col; colIdx++)
            {
                sum += isnan(x.at(rowIdx, colIdx)) ? 0 : x.at(rowIdx, colIdx);
            }
            res.data[rowIdx] = sum;
        }

        return res;
    }

    // Sum all elements in matrix
    double sumAllMatrix(Matrix x)
    {
        double sum = 0;
        for (int i = 0; i < x.col * x.row; i++)
            sum += isnan(x.data[i]) ? 0 : x.data[i];
        //cout << "sum = " << sum << endl;
        return sum;
    }

    Matrix squareAllMatrix(Matrix x)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            double tmp = res.data[i] * res.data[i];
            res.data[i] = isnan(tmp) ? 0 : tmp;
        }
        return res;
    }

    Matrix sqrtAllMatrix(Matrix x)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = sqrt(res.data[i]);
            //if (isnan(res.data[i])) res.data[i] = 0;
        }
        return res;
    }

    Matrix onesMatrix(int nrow, int ncol)
    {
        Matrix res;
        res.initSize(nrow, ncol);
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = 1;
            //if (isnan(res.data[i])) res.data[i] = 0;
        }
        return res;
    }

    Matrix divMatrixMatrix(Matrix x, Matrix a)
    {
        Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int i = 0; i < res.col * res.row; i++)
        {
            res.data[i] = x.data[i] / a.data[i];
            //if (isnan(res.data[i])) res.data[i] = 0;
        }
        return res;
    }

    Matrix normalize(Matrix x)
    {
        Matrix res = x;
        double max = x.data[0];
#pragma omp parallel for num_threads(NUM_THREAD)
        for (int colIdx = 0; colIdx < x.col; colIdx++)
        {
            //double max = x.at(0, colIdx);
            //double sum = 0;
            for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
            {
                if (max < x.at(rowIdx, colIdx)) max = x.at(rowIdx, colIdx);
                //sum += x.at(rowIdx, colIdx);
            }

            for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
            {
                res.data[rowIdx * res.col + colIdx] = x.data[rowIdx * res.col + colIdx] - max;
                //res.data[rowIdx * res.col + colIdx] /= sum;
                //res.data[rowIdx * res.col + colIdx] = x.data[rowIdx * res.col + colIdx] / 100;
            }
        }

        //for (int i = 0; i < x.col * x.row; i++)
        //{
        //    res.data[i] = x.data[i] / 100;
        //    res.data[i] = x.data[i] - max;
        //}
        //cout << "max isL " << max << endl;
        return res;
    }
}