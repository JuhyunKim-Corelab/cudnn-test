// cudNNTest.cpp : Defines the entry point for the console application.
/*
nvcc -I/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -L/home/seungbin/npu/tools/cudnn-6.5-linux-x64-v2 -lcudnn cudNNTest.cpp -o cudNNTest.exe
./cudNNTest.exe > result.target.test.data
vimdiff result.target.test.data target.test.data
*/
#include "cudnn.h"
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

#define EXIT_MSG(s) 								 \
do {                                                 \
    printf ("%s\n", s);                              \
    fflush (stdout);                                 \
    exit(1);										 \
} while (0)

float * readMatrix(char * filename, int nRows, int nCols);
void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor);

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

	char filename_img[1024] = "image.test.data";//"image.one.data";
	char filename_filter[1024] = "filter.test.data";//"filter.zero.data";
	char filename_targetInit[1024] = "targetInit.test.data";
	char filename_target[1024] = "target.test.data";

	cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
	int nDataTypeSize = (((int)dataType)+1) * sizeof(float); // = 4
	float *pImageInBatch_h = readMatrix(filename_img, c_in*h_in*w_in, n_in);
	float *pImageInBatch_d = NULL;
	float *pFilter_h = readMatrix(filename_filter, c_pFilter_in*h_pFilter_in*w_pFilter_in, k_pFilter_in);
	float *pFilter_d = NULL;

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
	err = cudaMemcpy(pImageInBatch_d, pImageInBatch_h, (size_t)(n_in*c_in*h_in*w_in*nDataTypeSize), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");
	err = cudaMalloc((void**)&pFilter_d , k_pFilter_in*c_pFilter_in*h_pFilter_in*w_pFilter_in * nDataTypeSize);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");
	err = cudaMemcpy(pFilter_d, pFilter_h, (size_t)(k_pFilter_in*c_pFilter_in*h_pFilter_in*w_pFilter_in * nDataTypeSize), cudaMemcpyHostToDevice);
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

	float* pImageOutBatch_h = readMatrix(filename_targetInit, c_out*h_out*w_out, n_out);
	float* pImageOutBatch_d = NULL;

	// Set and allocate output tensor descriptor
	status = cudnnSetTensor4dDescriptor(pOutputDesc, CUDNN_TENSOR_NCHW, dataType, n_out, c_out, h_out, w_out);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");
	err = cudaMalloc((void**)&pImageOutBatch_d, n_out*c_out*h_out*w_out * nDataTypeSize);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");
	err = cudaMemcpy(pImageOutBatch_d, pImageOutBatch_h, (size_t)(n_out*c_out*h_out*w_out * nDataTypeSize), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");

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
	//printf("workSpaceSizeInBytes: %ld\n", workSpaceSizeInBytes);

	float float_one_h = 1.0;

	/* Function to perform the forward multiconvolution */
	status = cudnnConvolutionForward(hCudNN, &float_one_h,
							  pInputDesc, pImageInBatch_d, pFilterDesc, pFilter_d, pConvDesc,
							  algo, workSpace_d, workSpaceSizeInBytes, &float_one_h,
                              pOutputDesc, pImageOutBatch_d);
	if (status != CUDNN_STATUS_SUCCESS) EXIT_MSG("ERROR..");

	err = cudaMemcpy(pImageOutBatch_h, pImageOutBatch_d, (size_t)(n_out*c_out*h_out*w_out * nDataTypeSize), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess) EXIT_MSG("ERROR ~");

	print_result(pImageOutBatch_h, c_out*h_out*w_out, n_out, c_out*h_out*w_out, n_out, 1);

	// Clean-up
	if (pImageInBatch_d != NULL)	cudaFree(pImageInBatch_d);
	if (pImageOutBatch_d != NULL)	cudaFree((void*)pImageOutBatch_d);
	if (pFilter_d != NULL)	cudaFree((void*)pFilter_d);
	if (workSpace_d != NULL)	cudaFree(workSpace_d);
	if (pInputDesc != NULL)	cudnnDestroyTensorDescriptor(pInputDesc);
	if (pOutputDesc != NULL)	cudnnDestroyTensorDescriptor(pOutputDesc);
	if (pFilterDesc != NULL)	cudnnDestroyFilterDescriptor(pFilterDesc);
	if (pConvDesc != NULL)	cudnnDestroyConvolutionDescriptor(pConvDesc);
	if (hCudNN != NULL)	cudnnDestroy(hCudNN);
	if (pImageInBatch_h != NULL)	free(pImageInBatch_h);
	if (pFilter_h != NULL)	free(pFilter_h);
	if (pImageOutBatch_h != NULL)	free(pImageOutBatch_h);

	return 0;
} 



float * readMatrix(char * filename, int nRows, int nCols)
{

    float tmp;
    FILE *fp;
    float *full;
    full = (float *) malloc (nRows*nCols*sizeof(full[0]));

    if((fp = fopen(filename, "r+")) == NULL) {
        printf("No such file: %s\n", filename);
        exit(1);
    }

    for (int i = 0; i < nRows; ++i)
    {
        for (int j = 0; j < nCols; ++j)
        {
            int ret = fscanf(fp, "%f ", &tmp);
            if(ret == 1){
                full[i*nCols + j] = tmp;
                //printf("%.15f\n", tmp);
            }
            else if(errno != 0) {
                    perror("scanf:");
                    break;
            } else if(ret == EOF) {
                //printf("finish.\n");
                break;
            } else {
                printf("No match.\n");
                exit(0);
            }
        }
    }

    return full;//full_dev
}



void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor){
    
    //printf("$$$$$$$$$ RESULT $$$$$$$$$$$$$$$\n");
    for (int y = 0; y < mR; ++y)
    {
        for (int x = 0; x < nR; ++x)
        {
            if(x<real_nR && y<real_mR){
                //if(result[nR*y + x] != -1.0)
                    printf("%.15f ", result[nR*y + x]);
            }
        }
        printf("\n");
    }
    //printf("============END==========\n");
}