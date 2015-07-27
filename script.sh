#!bin/sh


nvcc convOpti.cu -o convOpti.exe
time ./convOpti.exe > convOpti.target.test.data

nvcc -I/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -L/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -lcudnn cudNNTest.cu -o cudNNTest.exe
time ./cudNNTest.exe > result.target.test.data


cat -n convOpti.target.test.data > convOpti.target.test.n.data
cat -n result.target.test.data > result.target.test.n.data


vimdiff convOpti.target.test.n.data result.target.test.n.data


#
#time ./convOpti.exe > convOpti.target.test.data
#time ./cudNNTest.exe > result.target.test.data
#diff convOpti.target.test.data result.target.test.data
#