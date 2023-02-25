#include <iostream>
#include <random>
#include <chrono>
#include <omp.h>

using namespace std;

int main()
{
    cout << "start\n";
    const int NROWS = 10;
    const int NCOLS = 60000;    

    // Trans = (float*)malloc(sizeof(float)*NROWS*NCOLS);
    // Matrix = (float*)malloc(sizeof(float)*NROWS*NCOLS);

    float Matrix[10][60000];
    float Trans[10][60000];

    int maxLength = 10000;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    static std::default_random_engine generator(45);
    std::uniform_int_distribution<int> distribution(0, maxLength);

    int i,j;
    for (i = 0 ; i < NROWS ; i++)
    for (j = 0 ; j < NCOLS ; j++)
    Matrix[i][j] = distribution(generator) *1.0 / maxLength;

    auto startTime = std::chrono::steady_clock::now();  
    for (i = 0 ; i < NROWS ; i = i + 1)
    for (j = 0 ; j < NCOLS ; j = j + 1)
        Trans[j][i] = Matrix[i][j];

    auto endTime = std::chrono::steady_clock::now();
    auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    cout << "Total time for this function: " << encTime / 1000.0 << " sec." << endl;    

    return 0;
}