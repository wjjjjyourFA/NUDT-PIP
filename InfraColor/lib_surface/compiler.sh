#!/usr/bin/bash
#nullptr to NULL
nvcc -std=c++11 -arch=sm_61 -Xcompiler '-fPIC' -dc Getxyz.cu
nvcc -arch=sm_61 -Xcompiler '-fPIC' -dlink Getxyz.o
g++ -shared -o ./bin/Getxyz.so Getxyz.o a_dlink.o -L/usr/local/cuda/lib64 -lcudart -lcudnn -lcublas -lcudadevrt
rm *.o

nvcc -std=c++11 -arch=sm_61 -Xcompiler '-fPIC' -dc generator.cu
nvcc -arch=sm_61 -Xcompiler '-fPIC' -dlink generator.o
g++ -shared -o ./bin/generator.so generator.o a_dlink.o -L/usr/local/cuda/lib64 -lcudart -lcudnn -lcublas -lcudadevrt
rm *.o
#rm utils.so
