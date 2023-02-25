#include <fstream>
#include "matrix.h"
#include "readData.h"

#include <chrono>
#include <random>
#include <algorithm>
#include <time.h>
#include <ctime>

using namespace std;

class MLP
{
public:
    int numLayer;
    int numClass;
    int dataDim;
    int numData;
    vector <Matrix> Layer;
    vector <int> trainLabel;

    MLP(int _numClass, int _dataDim) { numClass = _numClass; dataDim = _dataDim; }

    void ReadData(Matrix& inputLayer);
    void InitParams();
    void Process();
    Matrix OneHotCoding(vector<int> label, int numClass);

private:
    vector <Matrix> W;
    vector <Matrix> b;

    vector <Matrix> dW;
    vector <Matrix> db;
    enum activationFunction
    {
        RELU = 0,
        SOFTMAX = 1
    };

    Matrix ForwardProp(Matrix input, Matrix weights, Matrix bias);
    void BackProp(Matrix input, Matrix output_gradient, Matrix& weighs, Matrix& bias, activationFunction af);
    void RandMatrix(Matrix& x);
    Matrix plusBias(Matrix x, Matrix b);


    void CheckAccuracy(Matrix outputLayer);
};

void MLP::RandMatrix(Matrix& x)
{
    int maxLength = 10000;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(45);
    std::uniform_int_distribution<int> distribution(0, maxLength);

    for (int i = 0; i < x.col * x.row; i++)
    {
        x.data[i] = distribution(generator) * 1.0 / maxLength - 0.5;
    }
}

Matrix MLP::plusBias(Matrix x, Matrix b)
{
    Matrix res = x;
    for (int colIdx = 0; colIdx < x.col; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
            res.data[rowIdx * res.col + colIdx] += b.data[rowIdx];
    }
    return res;
}



Matrix ReLU(Matrix x)
{
    Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < res.col * res.row; i++)
        res.data[i] = (res.data[i] > 0) ? res.data[i] : 0;

    return res;
}

Matrix deriv_ReLU(Matrix x)
{
    Matrix res = x;
#pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < res.col * res.row; i++)
        res.data[i] = (res.data[i] > 0) ? 1 : 0;

    return res;
}

Matrix softmax(Matrix x)
{
    Matrix res = x;
    vector <float> expSum;

    for (int colIdx = 0; colIdx < x.col; colIdx++)
    {
        float sum = 0;
        for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
        {
            sum += exp(x.at(rowIdx, colIdx));
        }
        expSum.push_back(sum);
        //cout << sum << endl;
    }
#pragma omp parallel for num_threads(NUM_THREAD)
    for (int colIdx = 0; colIdx < x.col; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < x.row; rowIdx++)
        {
            res.data[rowIdx * res.col + colIdx] = exp(res.at(rowIdx, colIdx)) / expSum[colIdx];
        }
    }

    return res;
}

Matrix MLP::OneHotCoding(vector<int> label, int numClass)
{
    int numLabel = label.size();
    Matrix res(numClass, numLabel);

    fill(res.data.begin(), res.data.end(), 0);
#pragma omp parallel for num_threads(NUM_THREAD)
    for (int i = 0; i < numLabel; i++)
    {
        int pos = label[i];
        res.data[pos * numLabel + i] = 1;
    }

    return res;
}

void MLP::InitParams()
{
    // 2-Layer MLP
    // List of layers: input layer, hidden layer 1, output layer

    // Define variables
    numData = 60000; //max 60000
    // dataDim = 784;
    // numClass = 10;

    // Initial number of layers
    int numNodesHiddenLayer = 10;
    int numLayer = 3;
    Matrix inputLayer(numData, dataDim);
    Matrix hiddenLayer1(numNodesHiddenLayer, numData);
    Matrix outputLayer(numClass, numData);

    // Initial weights and biases
    Matrix W1(numNodesHiddenLayer, dataDim);
    Matrix b1(numNodesHiddenLayer, 1);
    Matrix W2(numClass, numNodesHiddenLayer);
    Matrix b2(numClass, 1);
    RandMatrix(W1); RandMatrix(b1); RandMatrix(W2); RandMatrix(b2);
    W.push_back(W1); W.push_back(W2);
    b.push_back(b1); b.push_back(b2);

    //Read data from MNIST
    cout << "read data...\n";
    ReadDataMNIST(inputLayer, trainLabel, numData);
    cout << "add to layer...\n";
    Layer.push_back(matrix::transpose(inputLayer));
}

void MLP::Process()
{
    int i = 0;
    for (i = 0 ; i < 101 ; i++)
    {
        cout << "iteration: " << i << endl;


        ////////  Forward propagation    //////////

        Matrix Z1 = ForwardProp(Layer[0], W[0], b[0]);
        Matrix A1 = ReLU(Z1);
        Matrix Z2 = ForwardProp(A1, W[1], b[1]);
        Matrix A2 = softmax(Z2);
        
        ///////////////////////////////////////////


        ////////  Backward propagation    //////////

        Matrix Y = OneHotCoding(trainLabel, 10);
        Matrix dZ2 = matrix::matrix_minus(A2, Y);
        BackProp(A1, dZ2, W[1], b[1], SOFTMAX);
        Matrix dZ1 = matrix::mulAllMatrix(matrix::matrix_dot(W[1], dZ2), deriv_ReLU(Z1));
        BackProp(Layer[0], dZ1, W[0], b[0], RELU);

        ///////////////////////////////////////////


        //auto startTime = std::chrono::steady_clock::now();  
        //auto endTime = std::chrono::steady_clock::now();
        //auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        //cout << "Total time for this function: " << encTime / 1000.0 << " sec." << endl;

        if (i % 10 == 0)
        CheckAccuracy(A2);
    }

}

Matrix MLP::ForwardProp(Matrix input, Matrix weights, Matrix bias)
{
    return plusBias(matrix::matrix_dot(weights, input), bias);
}

void MLP::BackProp(Matrix input, Matrix output_gradient, Matrix& weights, Matrix& bias, activationFunction af)
{
    float learning_rate = 0.1;
    Matrix weights_gradient = matrix::matrix_dot(output_gradient, matrix::transpose(input));
    weights_gradient = matrix::divAll(weights_gradient, numData * 1.0);
    float bias_gradient = matrix::sumAllMatrix(output_gradient) * 1.0 / numData;
    weights = matrix::matrix_minus(weights, matrix::mulAll(weights_gradient, learning_rate));
    bias = matrix::minusAll(bias, learning_rate*bias_gradient);
}

int convertToLabel(vector<float>data)
{
    int pos = 0;
    float max = data[0];
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

void MLP::CheckAccuracy(Matrix outputLayer)
{
    // MNIST output layer should be: 10 x numData
    float totalAcc = 0;
    for (int colIdx = 0; colIdx < outputLayer.col; colIdx++)
    {
        vector <float> data;
        for (int rowIdx = 0; rowIdx < outputLayer.row; rowIdx++)
        {
            data.push_back(outputLayer.at(rowIdx, colIdx));
        }
        totalAcc += (trainLabel[colIdx] == convertToLabel(data));
    }
    cout << "Accuracy: " << totalAcc << " over " << numData << ", " << totalAcc / numData <<endl;
}

int main()
{
    /////// Test ///////

    // Matrix A(3,2); 
    // A.data[0] = 1; A.data[1] = 2; 
    // A.data[2] = 3; A.data[3] = 4;
    // A.data[4] = 5; A.data[5] = 6;
    // matrix::showMatrix(A);

    // Matrix B(2,3);
    // B.data[0] = 1; B.data[1] = 2; B.data[2] = 3; 
    // B.data[3] = 4; B.data[4] = 5; B.data[5] = 6;
    // matrix::showMatrix(B);

    // Matrix res = matrix::matrix_dot(B,A);
    // matrix::showMatrix(res);

    ////////////////////

    MLP mnist(10, 784);
    mnist.InitParams();
    mnist.Process();
    cout << "program exited!" << endl;
    return 0;
}