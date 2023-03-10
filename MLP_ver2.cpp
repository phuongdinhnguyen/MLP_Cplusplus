#include <fstream>
#include "matrix.h"
#include "readData.h"
#include "utils.h"


#include <chrono>
#include <random>
#include <algorithm>
#include <time.h>
#include <ctime>

using namespace std;

Matrix plusBias(Matrix x, Matrix b)
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
//#pragma omp parallel for num_threads(NUM_THREAD)
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

Matrix Softmax(Matrix x)
{ 
    Matrix norm_x = matrix::normalize(x);
    Matrix res = x;
    vector <double> expSum;

    for (int colIdx = 0; colIdx < norm_x.col; colIdx++)
    {
        double sum = 0;
        for (int rowIdx = 0; rowIdx < norm_x.row; rowIdx++)
        {
            //double tmp = exp(norm_x.at(rowIdx, colIdx));
            sum += exp(norm_x.at(rowIdx, colIdx));
        }
        expSum.push_back(sum);
        //cout << sum << endl;
    }
//#pragma omp parallel for num_threads(NUM_THREAD)
    for (int colIdx = 0; colIdx < norm_x.col; colIdx++)
    {
        for (int rowIdx = 0; rowIdx < norm_x.row; rowIdx++)
        {
            res.data[rowIdx * norm_x.col + colIdx] = exp(norm_x.at(rowIdx, colIdx)) / expSum[colIdx];      
            if (isnan(res.data[rowIdx * norm_x.col + colIdx]))
            {
                cout << "x=" << norm_x.at(rowIdx, colIdx) << endl;
                cout << "sum=" << expSum[colIdx] << endl;
            }

        }
    }
    
    return res;
}

enum ActivationFunction
{
    AF_none = -1,
    AF_ReLU = 0,
    AF_Softmax = 1
};

enum optimizer
{
    GD = 0,
    GD_MOMENTUM = 1,
    RMSPROP = 2
};

class Layer
{
public:
    int numNodes;
    int numInput;
    int numData;
    int layerPos;

    ActivationFunction AF;
    Layer(){}
    Layer(int _numInput, int _numNodes, ActivationFunction _af)
    {numInput = _numInput; numNodes = _numNodes; AF = _af;}

    void setLayerPos(int _layerPos){layerPos = _layerPos; }

    Matrix W, b, A, Z;
    Matrix dZ, dW, db, dA;
    Matrix prev_dW;
    Matrix prev_db;

    Matrix prev_v_dW; // for RMSProp
    Matrix prev_v_db;

    // If this is first layer -> input layer
    void setInput(Matrix inputLayer)
        {A = matrix::transpose(inputLayer);}

    void Init()
    {
        // prepare weights and bias
        if (layerPos == 0) return;
        W.initSize(numNodes, numInput);
        b.initSize(numNodes, 1);
        matrix::RandMatrix(W);
        //matrix::RandMatrix(b);

        prev_v_dW.initSize(numNodes, numInput);
        prev_v_db.initSize(numNodes, 1);
        prev_dW.initSize(numNodes, numInput);
        prev_db.initSize(numNodes, 1);
    }

    void Forward(Layer prevLayer)
    {
        /////////// Algorithm for Forward propagation ////////////
        // 
        // Z = WX + b
        // A = ActivationFunction(Z)
        // 
        //////////////////////////////////////////////////////////

        Matrix WdotX = matrix::matrix_dot(W, prevLayer.A);
        Z = plusBias(WdotX, b);
        if (AF == AF_ReLU)
            A = ReLU(Z);
        else if (AF == AF_Softmax)
            A = Softmax(Z);
    }

    void Backward(Layer prevLayer, Layer nextLayer, vector <int> label)
    {
        /////////// Algorithm for Backward propagation ////////////
        // 
        // for Softmax: dZ = A - Y
        // for ReLU:    dZ = W dot dZ * deriv_ActivationFunction(Z)
        // dW = dZ dot prevA / numData
        // dB = sum(dZ) / numData
        // 
        ///////////////////////////////////////////////////////////

        if (AF == AF_Softmax)
        {
            Matrix oneHotY = OneHotCoding(label, 10);
            dZ = matrix::matrix_minus(A, oneHotY);
        }
        else if (AF == AF_ReLU)
        {
            dZ = matrix::mulAllMatrix(nextLayer.dA,deriv_ReLU(Z));
        }

        db = matrix::sumRow(dZ);
        //db = matrix::divAll(db, numData);

        Matrix prevA_T = matrix::transpose(prevLayer.A);
        dW = matrix::matrix_dot(dZ, prevA_T);
        //dW = matrix::divAll(dW, numData);

        Matrix W_T = matrix::transpose(W);
        dA = matrix::matrix_dot(W_T, dZ);
    }
  
    void update(double learningRate, optimizer opt)
    {
        // SGD-momentum
        
        // Update weights and bias, with learning rate
        if (opt == GD || opt == GD_MOMENTUM)
        {
            if (opt == GD)
            {
                //cout << "Running GD: " << endl;
                W = matrix::matrix_minus(W, matrix::mulAll(dW, learningRate));
                b = matrix::matrix_minus(b, matrix::mulAll(db, learningRate));
            }

            double momentum = 0.9;
            if (opt == GD_MOMENTUM)
            {
                prev_dW = matrix::matrix_minus(matrix::mulAll(prev_dW, momentum), matrix::mulAll(dW, learningRate));
                prev_db = matrix::matrix_minus(matrix::mulAll(prev_db, momentum), matrix::mulAll(db, learningRate));

                W = matrix::matrix_plus(W, prev_dW);
                W = matrix::matrix_plus(b, prev_db);
            }
        }

        // RMSProp
        if (opt == RMSPROP)
        {
            double beta = 0.9;
            double epsilon = 10e-8;
            double alpha = 1e-4;
            Matrix beta_v_dw = matrix::mulAll(prev_v_dW, beta);
            Matrix dW_square = matrix::squareAllMatrix(dW);
            Matrix v_dW = matrix::matrix_plus(beta_v_dw, matrix::mulAll(dW_square, (1 - beta)));

            Matrix beta_v_db = matrix::mulAll(prev_v_db, beta);
            Matrix db_square = matrix::squareAllMatrix(db);
            Matrix v_db = matrix::matrix_plus(beta_v_db, matrix::mulAll(db_square, (1 - beta)));

            Matrix tmpCalc = matrix::divMatrixMatrix(dW, matrix::plusAll(matrix::sqrtAllMatrix(v_dW), epsilon));

            W = matrix::matrix_minus(W, matrix::mulAll(tmpCalc, alpha));

            Matrix tmpCalc2 = matrix::divMatrixMatrix(db, matrix::plusAll(matrix::sqrtAllMatrix(v_db), epsilon));
            b = matrix::matrix_minus(b, matrix::mulAll(tmpCalc2, alpha));
        }
        
    }
};

class InputLayer : public Layer
{
public:
    // Seperate Layer definition for first layer, which is input layer
    InputLayer(int _numData, int _dim, Matrix inputLayer)
    {
        //A = matrix::transpose(inputLayer);
        A = inputLayer;
        numData = _numData;
    }
};
ofstream file("output.txt");
class Network
{
public:
    int numEpoch;
    double learningRate;
    Matrix inputLayer;

    Matrix validateData;
    vector <int> validateLabel;

    Matrix testData;
    vector <int> testLabel;

    int batchSize;
        
    vector <double> _inputLayer;

    int applySGD = 1;
    int dataDim;

    vector <Layer> layer;
    vector <int> label;

    int numLayer(){return layer.size();}
    void setDataInputDim(int _inputDim) { dataDim = _inputDim; }
    void setEpoch(int _numEpoch) { numEpoch = _numEpoch; }
    void setLearningRate(double _learningRate) { learningRate = _learningRate; }
    void setBatchSize(int _batchSize) { batchSize = _batchSize; }

    void addLayer(Layer _layer)
    { 
        int numLayer = layer.size();
        _layer.setLayerPos(numLayer);
        layer.push_back(_layer); 
    }

    void initParams(int doVal)
    {   
        // Init all parameters: weights, bias
        for (int i = 1 ; i < numLayer() ; i++)
            layer[i].Init();
        
        // Push a null layer (after outputLayer), because of my own algorithm :')
        Layer nullLayer;
        layer.push_back(nullLayer);

        // Extract validation
        if (doVal)
        {
            int numValData = 1000;
            // extract validate data (pics)
            validateData.data.assign(inputLayer.data.end() - numValData * dataDim, inputLayer.data.end());
            validateData.row = numValData;
            validateData.col = dataDim;

            // delete last numbers of data already being used for validate
            int numInputData = layer[0].numData;
            inputLayer.data.resize((numInputData - numValData) * dataDim);
            inputLayer.row = numInputData - numValData;

            // edit first layer ~ input layer:
            layer[0].numData = numInputData - numValData;
            layer[0].A = inputLayer;

            // extract validate label
            validateLabel.assign(label.end() - numValData, label.end());

            // delete last numbers of labels already being used for validate
            label.resize(numInputData - numValData);
        }
    }

    void updateParams(double learningRate, optimizer opt)
    {
        for (int i = 1; i < numLayer() - 1; i++)
        {
            layer[i].update(learningRate, opt);
        }
    }

    double loss(Layer outputLayer, vector <int> _label)
    {
        Matrix Y = OneHotCoding(_label, 10);
        double sum = 0;
        int size = Y.data.size();
        for (int i = 0; i < Y.data.size(); i++)
        {
            sum += Y.data[i] * log(outputLayer.A.data[i]);
        }
        double res = -sum / size;

        return res;
    }

    void fit()
    { 
        int posOutputLayer = numLayer() - 2;
        int inputDim = inputLayer.col;
        Layer originalInputLayer = layer[0];
        double trainingAcc = 0;
        cout << "Number of data: " << originalInputLayer.numData << endl;
        for (int epoch = 0 ; epoch < numEpoch ; epoch++)
        {
            trainingAcc = 0;
            cout << "(Epoch " << epoch << " / " << numEpoch - 1 << ")" << endl;
            file << "(Epoch " << epoch << " / " << numEpoch - 1 << ")" << endl;
            //file << "epoch: " << epoch << endl;
            //int batch_size = originalInputLayer.numData;
            int batch_size = 100;
            double totalLoss = 0;
            int numBatch = Network::label.size() / batch_size;
            for (int batchIdx = 0; batchIdx < numBatch; batchIdx++)
            {
                double batchLoss = 0;
                // Applying minibatch or batch or SGD: edit first layer
                Matrix x;
                x.data.assign(inputLayer.data.begin() + batchIdx * batch_size * inputDim, 
                              inputLayer.data.begin() + (batchIdx + 1) * batch_size * inputDim); // batch data
                x.col = inputDim;
                x.row = batch_size;
                layer[0].A = matrix::transpose(x);
                layer[0].numData = batch_size;

                vector <int> label; //batch label
                label.assign(Network::label.begin() + batchIdx * batch_size, Network::label.begin() + (batchIdx + 1) * batch_size);

                // Forward propagation with Activation Function
                for (int i = 0; i < numLayer() - 2; i++)
                {
                    layer[i + 1].Forward(layer[i]);
                }

                // Backward propagation with deriverate of Activation Function
                for (int i = numLayer() - 2; i > 0; i--)
                {
                    layer[i].Backward(layer[i - 1], layer[i + 1], label);
                }

                // Update parameters
                updateParams(learningRate, RMSPROP);

                batchLoss = loss(layer[posOutputLayer], label);
                double acc = CheckAccuracy(layer[posOutputLayer].A, label, batch_size);
                trainingAcc += acc;
                //if (batchIdx % (numBatch / 10) == 0)
                {
                    cout << "(Iteration " << batchIdx << " / " << numBatch << ") loss:" << batchLoss << endl;
                    file << "(Iteration " << batchIdx << " / " << numBatch << ") loss:" << batchLoss << endl;
                }
            }
      
            ////////////////// train accuracy/////////////////////////
            cout << "training accuracy: " << trainingAcc / Network::label.size() << endl;
            file << "training accuracy: " << trainingAcc / Network::label.size() << endl;

            ////////////////// val accuracy  ///////////////////////////
            layer[0].A = matrix::transpose(validateData);
            for (int i = 0; i < numLayer() - 2; i++)
            {
                layer[i + 1].Forward(layer[i]);
            }
            double valAcc = CheckAccuracy(layer[posOutputLayer].A, validateLabel, validateData.row);
            cout << "validate accuracy: " << valAcc / validateData.row << endl;
            file << "validate accuracy: " << valAcc / validateData.row << endl;

            cout << "\n";
        }

        readTestData();
        cout << " test ";
        layer[0].A = matrix::transpose(testData);
        for (int i = 0; i < numLayer() - 2; i++)
        {
            layer[i + 1].Forward(layer[i]);
        }
        double valAcc = CheckAccuracy(layer[posOutputLayer].A, testLabel, testData.row);
        file << "test accuracy: " << valAcc << " over " << 10000 << endl;
        cout << "test accuracy: " << valAcc << " over " << 10000 << endl;
        cout << "\n";
        
    }

    void predict()
    {
        
    }

    void readData(int numFiles, int numDataEachFile); //cifar10 only
    void readTestData(); //cifar10 only
    // divide validation
};

int main()
{
    // TEST FUNCTION
    //testingAllFunctions();
      
    // Define new layer: Layer(number of input, number of layer's nodes, activation function)

    Network MLP;
    //int numData = 5000;
    //int dataDim = 784;
    //ReadDataMNIST(MLP.inputLayer, MLP.label, numData);
    //MLP.setEpoch(50);
    //MLP.setDataInputDim(dataDim);
    //MLP.addLayer(InputLayer(numData,dataDim,MLP.inputLayer));
    //MLP.addLayer(Layer(784, 100, AF_ReLU));
    //MLP.addLayer(Layer(100, 10, AF_Softmax));
    //MLP.initParams(1);
    //MLP.setLearningRate(0.01);
    //MLP.fit();

    
    int numDataEachFile = 10000;
    int numFiles = 5;
    int numData = numFiles * numDataEachFile;
    int dataDim = 3072;
    MLP.readData(numFiles, numDataEachFile);
    MLP.setEpoch(6);
    MLP.setDataInputDim(dataDim);
    MLP.addLayer(InputLayer(numData, dataDim, MLP.inputLayer));
    MLP.addLayer(Layer(dataDim, 200, AF_ReLU));
    //MLP.addLayer(Layer(100, 100, AF_ReLU));
    //MLP.addLayer(Layer(100, 100, AF_ReLU));
    //MLP.addLayer(Layer(100, 100, AF_ReLU));
    //MLP.addLayer(Layer(100, 100, AF_ReLU));
    MLP.addLayer(Layer(200, 10, AF_Softmax));
    MLP.initParams(1);
    MLP.setLearningRate(1e-3);
    MLP.fit();
    
    cout << "program exited!" << endl;
    return 0;
}

void Network::readData(int numFiles, int numDataEachFile)
{
    int numberOfData = numDataEachFile;
    int numberOfFile = numFiles;
    int dim = 3072;
    ifstream file3;

    inputLayer.data.resize(numFiles * numDataEachFile * dim);
    //inputLayer.data.resize(0);
    inputLayer.row = numFiles * numDataEachFile;
    inputLayer.col = dim;
    //vector <double> _inputLayer;
    //vector <int> label;
    int dataPos = 0;
    for (int file_idx = 0; file_idx < numberOfFile; file_idx++)
    {
        file3.open(CIFAR10_file[file_idx], ios::binary);
        if (file3.is_open())
        {
            cout << "open and reading traing file... \n";
            struct data;
            unsigned char _label = 0;
            for (int cnt = 0; cnt < numberOfData; cnt++)
            {
                file3.read((char*)&_label, sizeof(_label));
                label.push_back((int)_label);
                for (int i = 0; i < dim; ++i)
                {
                    unsigned char pixel = 0;
                    //int dataPos = file_idx * numDataEachFile * dim + cnt * numberOfData + i;
                    file3.read((char*)&pixel, sizeof(pixel));
                    //inputLayer.data.push_back((double)pixel / 255.0);
                    
                    inputLayer.data[dataPos] = (double)pixel / 255.0;
                    dataPos++;

                    //_inputLayer.push_back((double)pixel / 255.0);
                }
            }
        }
        file3.close();
        cout << "done 1 file! \n";
    }
    matrix::checkMatrix(inputLayer);
    cout << inputLayer.data.size() << endl;
}

void Network::readTestData()
{
    int numberOfData = 10000;
    int dim = 3072;
    ifstream file3;

    testData.data.resize(numberOfData * dim);
    testData.row = numberOfData;
    testData.col = dim;

    int cnt = 0;
    file3.open(CIFAR10_test, ios::binary);
    if (file3.is_open())
    {
        cout << "open and reading test data... \n";
        unsigned char label = 0;
        for (int i = 0; i < numberOfData; i++)
        {
            file3.read((char*)&label, sizeof(label));
            testLabel.push_back((int)label);
            for (int i = 0; i < dim; ++i)
            {
                unsigned char pixel = 0;
                file3.read((char*)&pixel, sizeof(pixel));
                //data.push_back((double)pixel / 255.0);
                testData.data[cnt] = (double)pixel / 255.0;
            }
        }
        cout << "done reading test data!\n";
    }
    file3.close();
}