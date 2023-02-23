#include <iostream>
#include <chrono>
#include <omp.h>

#define ROUND_UP(x, s) (((x)+((s)-1)) & -(s))

using namespace std;

inline void transpose_scalar_block(float *A, float *B, const int lda, const int ldb, const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<block_size; i++) {
        for(int j=0; j<block_size; j++) {
            B[j*ldb + i] = A[i*lda +j];
        }
    }
}

inline void transpose_block(float *A, float *B, const int n, const int m, const int lda, const int ldb, const int block_size) {
    #pragma omp parallel for
    for(int i=0; i<n; i+=block_size) {
        for(int j=0; j<m; j+=block_size) {
            transpose_scalar_block(&A[i*lda +j], &B[j*ldb + i], lda, ldb, block_size);
        }
    }
}

int main()
{
    const int n = 2;
    const int m = 3;
    int lda = ROUND_UP(m, 16);
    int ldb = ROUND_UP(n, 16);

    cout << "lda = " << lda << endl;
    cout << "ldb = " << ldb << endl;

    float *A = (float*)_mm_malloc(sizeof(float)*lda*ldb, 64);
    float *B = (float*)_mm_malloc(sizeof(float)*lda*ldb, 64);

    A[0] = 1; A[1] = 2; A[2] = 3;
    A[3] = 4; A[4] = 5; A[5] = 6;

    auto startTime = std::chrono::steady_clock::now();
    transpose_block(A,B,n,m,lda,ldb,16);
    auto endTime = std::chrono::steady_clock::now();
    auto encTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

    cout << "Total run time: " << encTime / 1000.0 << " sec." << endl;
    
    for (int i = 0 ; i < 64 ; i++)
    {
        if (i%15==0) cout << endl;
        cout << B[i] << " ";
    }
    cout << endl;
    return 0;
}