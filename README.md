# MLP_Cplusplus
Simple MLP Network in Vanilla C++

## Description

A program using only vanilla C++ libraries to create a simple Neural Network, as many layers 
and input as you like.

## Getting Started

### Dependencies

* Compiler (with OpenMP support). I'm using MSVC (Visual Studio) and MinGW 
* Prefer working on Visual Studio, easier to debug
* Dataset (currently using CIFAR-10, and MNIST)

### Executing program

* Compile and run:
```
g++ -fopenmp -O2 MLP_ver2.cpp -o MLP
.\MLP.exe
```

## Bugs/Problems

* This program is not optimize for large datasets. Program may exit due to lack of memory.
* Needs multithreading to improve running time.

## Results

* Running with 50000 datas: 49000 for training, 1000 for validating.
![alt text](https://github.com/phuongdinhnguyen/MLP_Cplusplus/blob/main/results/Visualize_results.PNG?raw=false)

## Authors

Me and my friends. From HUST IoT K64
* [@PhuongDinh](https://github.com/phuongdinhnguyen)
* [@ThaoAnh](https://github.com/aquarter147)
* [@KienLe]()

