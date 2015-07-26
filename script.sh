#!bin/sh


nvcc convOpti.cu -o convOpti.exe
./convOpti.exe > convOpti.target.test.data

nvcc -I/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -L/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -lcudnn cudNNTest.cpp -o cudNNTest.exe
./cudNNTest.exe > result.target.test.data


cat -n convOpti.target.test.data > convOpti.target.test.n.data
cat -n result.target.test.data > result.target.test.n.data


vimdiff convOpti.target.test.n.data result.target.test.n.data

