
// cudNNTest.cpp : Defines the entry point for the console application.
//
// Warning: Use at your own risk.

#include "stdafx.h"
#include "<your path here>\cudnn-6.5-win-R1\cudnn-6.5-win-R1\cudnn.h"


int _tmain(int argc, _TCHAR* argv[])
{
cudnnHandle_t hCudNN = NULL;
cudnnTensor4dDescriptor_t pInputDesc = NULL;
cudnnFilterDescriptor_t pFilterDesc = NULL;
cudnnConvolutionDescriptor_t pConvDesc = NULL;
cudnnTensor4dDescriptor_t pOutputDesc = NULL;
cudnnStatus_t status;
cudaError_t err;
int n_in = 64; // Number of images - originally 128
int c_in = 96; // Number of feature maps per image - originally 96
int h_in = 221; // Height of each feature map - originally 221
int w_in = 221; // Width of each feature map - originally 221
int k_pFilter_in = 256; // Number of output feature maps - originally 256
int c_pFilter_in = c_in; // Number of input feature maps - originally 96
int h_pFilter_in = 7; // Height of each pFilter - originally 7
int w_pFilter_in = 7; // Width of each pFilter - originally 7
int n_out = 0; // Number of output images.
int c_out = 0; // Number of output feature maps per image.
int h_out = 0; // Height of each output feature map.
int w_out = 0; // Width of each output feature map.

/* to change to double, chance CUDNN_DATA_FLOAT to CUDNN_DATA_DOUBLE and change each float to double below */

cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
int nDataTypeSize = (((int)dataType)+1) * sizeof(float);
float* pImageInBatch = NULL;
float* pFilter = NULL;
float* pImageOutBatch = NULL;


try
{
//---------------------------------------
// Create CudNN
//---------------------------------------

status = cudnnCreate(&hCudNN);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


//---------------------------------------
// Create Descriptors
//---------------------------------------

status = cudnnCreateTensor4dDescriptor(&pInputDesc);
if (status != CUDNN_STATUS_SUCCESS)
throw status;

status = cudnnCreateTensor4dDescriptor(&pOutputDesc);
if (status != CUDNN_STATUS_SUCCESS)
throw status;

status = cudnnCreateFilterDescriptor(&pFilterDesc);
if (status != CUDNN_STATUS_SUCCESS)
throw status;

status = cudnnCreateConvolutionDescriptor(&pConvDesc);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


//---------------------------------------
// Allocate memory for pFilter and ImageBatch
//---------------------------------------

err = cudaMalloc((void**)&pImageInBatch, n_in*c_in*h_in*w_in * nDataTypeSize);
if (err != cudaSuccess)
throw err;

err = cudaMalloc((void**)&pFilter , k_pFilter_in*c_pFilter_in*h_pFilter_in*w_pFilter_in * nDataTypeSize);
if (err != cudaSuccess)
throw err;


//---------------------------------------
// Fill the input image and pFilter data
//---------------------------------------

//TODO: Still figuring this out


//---------------------------------------
// Set decriptors
//---------------------------------------

status = cudnnSetTensor4dDescriptor(pInputDesc, CUDNN_TENSOR_NCHW, dataType, n_in, c_in, h_in, w_in);
if (status != CUDNN_STATUS_SUCCESS)
throw status;

status = cudnnSetFilterDescriptor(pFilterDesc, dataType, k_pFilter_in, c_pFilter_in, h_pFilter_in, w_pFilter_in);
if (status != CUDNN_STATUS_SUCCESS)
throw status;

status = cudnnSetConvolutionDescriptor(pConvDesc, pInputDesc, pFilterDesc, 0, 0, 2, 2, 1, 1, CUDNN_CONVOLUTION);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


//---------------------------------------
// Query output layout
//---------------------------------------

status = cudnnGetOutputTensor4dDim(pConvDesc, CUDNN_CONVOLUTION_FWD, &n_out, &c_out, &h_out, &w_out);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


//---------------------------------------
// Set and allocate output tensor descriptor
//---------------------------------------

status = cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, dataType, n_out, c_out, h_out, w_out);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


err = cudaMalloc((void**)&pImageOutBatch, n_out*c_out*h_out*w_out * nDataTypeSize);
if (err != cudaSuccess)
throw err;


//---------------------------------------
// Launch convolution on GPU
//---------------------------------------

status = cudnnConvolutionForward(hCudNN, pInputDesc, pImageInBatch, pFilterDesc, pFilter, pConvDesc, pOutputDesc, pImageOutBatch, CUDNN_RESULT_NO_ACCUMULATE);
if (status != CUDNN_STATUS_SUCCESS)
throw status;


//---------------------------------------
// Extract output data
//---------------------------------------

//TBD
}
catch (...)
{
}

//---------------------------------------
// Clean-up
//---------------------------------------

if (pImageInBatch != NULL)
cudaFree(pImageInBatch);

if (pImageOutBatch != NULL)
cudaFree((void*)pImageOutBatch);

if (pFilter != NULL)
cudaFree((void*)pFilter);

if (pInputDesc != NULL)
cudnnDestroyTensor4dDescriptor(pInputDesc);

if (pOutputDesc != NULL)
cudnnDestroyTensor4dDescriptor(pOutputDesc);

if (pFilterDesc != NULL)
cudnnDestroyFilterDescriptor(pFilterDesc);

if (pConvDesc != NULL)
cudnnDestroyConvolutionDescriptor(pConvDesc);

if (hCudNN != NULL)
cudnnDestroy(hCudNN);

return 0;
} 