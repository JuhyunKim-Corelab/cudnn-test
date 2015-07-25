// cudNNTest.cpp : Defines the entry point for the console application.
//nvcc -I/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -L/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -lcudnn cudNNTest.cpp -o cudNNTest.exe
#include "cudnn.h"
#include <stdio.h>
#include <stdlib.h>

#define EXIT_MSG(s) 								 \
do {                                                 \
    printf ("%s\n", s);                              \
    fflush (stdout);                                 \
    exit(1);										 \
} while (0)

int main(int argc, char* argv[])
{
	cudnnHandle_t hCudNN = NULL;
	cudnnTensorDescriptor_t pInputDesc = NULL;
	cudnnFilterDescriptor_t pFilterDesc = NULL;
	cudnnTensorDescriptor_t pOutputDesc = NULL;
	cudnnConvolutionDescriptor_t pConvDesc = NULL;
	cudnnStatus_t status;
	cudaError_t err;

	int n_in = 1; // Number of images - originally 128
	int c_in = 64; // Number of feature maps per image - originally 96
	int h_in = 12; // Height of each feature map - originally 221
	int w_in = 12; // Width of each feature map - originally 221

	int k_pFilter_in = 64; // Number of output feature maps //number of filters
	int c_pFilter_in = c_in; // Number of input feature maps - originally 96
	int h_pFilter_in = 5; // Height of each pFilter - originally 7
	int w_pFilter_in = 5; // Width of each pFilter - originally 7

	int n_out;// = 1; // Number of output images.
	int c_out;// = 64; // Number of output feature maps per image.
	int h_out;// = 12; // Height of each output feature map.
	int w_out;// = 12; // Width of each output feature map.

	/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */

	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	int nDataTypeSize = (((int)dataType)+1) * sizeof(float);
	float* pImageInBatch_d = NULL;
	float* pFilter_d = NULL;
	float* pImageOutBatch_d = NULL;

	// Create CudNN
	status = cudnnCreate(&hCudNN);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR.. cudnnCreate");

	// Create Descriptors
	status = cudnnCreateTensorDescriptor(&pInputDesc);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	status = cudnnCreateTensorDescriptor(&pOutputDesc);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	status = cudnnCreateFilterDescriptor(&pFilterDesc);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	status = cudnnCreateConvolutionDescriptor(&pConvDesc);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");

	// Allocate memory for pFilter and ImageBatch
	err = cudaMalloc((void**)&pImageInBatch_d, n_in*c_in*h_in*w_in * nDataTypeSize);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");
	err = cudaMalloc((void**)&pFilter_d , k_pFilter_in*c_pFilter_in*h_pFilter_in*w_pFilter_in * nDataTypeSize);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");

	
	// Fill the input image and pFilter data
	//TODO: Still figuring this out
	// Set decriptors
	status = cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, dataType, n_in, c_in, h_in, w_in);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	status = cudnnSetFilter4dDescriptor(pFilterDesc, dataType, k_pFilter_in, c_pFilter_in, h_pFilter_in, w_pFilter_in);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	status = cudnnSetConvolution2dDescriptor(pConvDesc, 2, 2, 1, 1, 1, 1, CUDNN_CONVOLUTION);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");

	/* Helper function to return the dimensions of the output tensor given a convolution descriptor */
	status = CUDNNWINAPI cudnnGetConvolution2dForwardOutputDim( pConvDesc, pInputDesc, pFilterDesc, &n_out, &c_out, &h_out, &w_out);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	//printf("n_out:%d, c_out:%d, h_out:%d, w_out:%d\n", n_out, c_out, h_out, w_out);

	// Set and allocate output tensor descriptor
	status = cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, dataType, n_out, c_out, h_out, w_out);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	err = cudaMalloc((void**)&pImageOutBatch_d, n_out*c_out*h_out*w_out * nDataTypeSize);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");

	// Launch convolution on GPU
	//status = cudnnConvolutionForward(hCudNN, pInputDesc, pImageInBatch_d, pFilterDesc, pFilter,
	//								 pConvDesc, pOutputDesc, pImageOutBatch_d, CUDNN_RESULT_NO_ACCUMULATE);
	//if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");

	//CUDNN_CONVOLUTION_FWD_NO_WORKSPACE
	//CUDNN_CONVOLUTION_FWD_PREFER_FASTEST
	//CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT
	size_t memoryLimitInbytes = 0;
	cudnnConvolutionFwdAlgo_t algo;
	status = cudnnGetConvolutionForwardAlgorithm(hCudNN, pInputDesc, pFilterDesc, pConvDesc, pOutputDesc,
											     CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, memoryLimitInbytes, &algo);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");

	size_t workSpaceSizeInBytes = 0;
	void *workSpace_d = NULL;
	status = cudnnGetConvolutionForwardWorkspaceSize(hCudNN, pInputDesc, pFilterDesc, pConvDesc, pOutputDesc,
													 algo, &workSpaceSizeInBytes);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	err = cudaMalloc((void**)&workSpace_d, workSpaceSizeInBytes);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");
	printf("workSpaceSizeInBytes: %ld\n", workSpaceSizeInBytes);

	float float_one_h = 1.0;

/* Function to perform the forward multiconvolution */

	status = cudnnConvolutionForward(hCudNN, &float_one_h,
								  pInputDesc, pImageInBatch_d, pFilterDesc, pFilter_d, pConvDesc,
								  algo, workSpace_d, workSpaceSizeInBytes, &float_one_h,
                                  pOutputDesc, pImageOutBatch_d);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");


	// Clean-up
	if (pImageInBatch_d != NULL)
	cudaFree(pImageInBatch_d);
	if (pImageOutBatch_d != NULL)
	cudaFree((void*)pImageOutBatch_d);
	if (pFilter_d != NULL)
	cudaFree((void*)pFilter_d);
	if (pInputDesc != NULL)
	cudnnDestroyTensorDescriptor(pInputDesc);
	if (pOutputDesc != NULL)
	cudnnDestroyTensorDescriptor(pOutputDesc);
	if (pFilterDesc != NULL)
	cudnnDestroyFilterDescriptor(pFilterDesc);
	if (pConvDesc != NULL)
	cudnnDestroyConvolutionDescriptor(pConvDesc);
	if (hCudNN != NULL)
	cudnnDestroy(hCudNN);
	
	return 0;
} 