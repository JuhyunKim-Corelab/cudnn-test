//convOpti.cu

/*
nvcc convOpti.cu -o convOpti.exe
*/

//#include <convOpti.cuh>
//#include <nvmatrix.cuh>
#include <string.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <assert.h>

#ifndef N_LIVING_NEURON
#define N_LIVING_NEURON 2027 // it's just for compiler option
#endif
#ifndef GRID_DIM_X
#define GRID_DIM_X 64 // it's just for compiler option
#endif
#define BLOCK_SIZE 32

#define EXIT_MSG(s) 								 \
do {                                                 \
    printf ("%s\n", s);                              \
    fflush (stdout);                                 \
    exit(1);										 \
} while (0)

float * readMatrix(char * filename, int nRows, int nCols);
void print_result(float* result, int mR, int nR, int real_mR, int real_nR, int isRowMajor);
template <int T_numImages>
__global__ void optiConvFilter(float* images, float* filters, float* targets,
                                   const int numImages, const int numFilters, //128(1), 64
                                   const int imgSizeY, const int imgSizeX, //12, 12
                                   const int filterSize, const int paddingStart, //5, -2
                                   const int moduleStride, //1
                                   const int numModulesY, const int numModulesX, //12, 12
                                   const int imgStride, const int numImgColors, //128(1), 64
                                   const float scaleTargets, const float scaleOutputs, //0.0, 1.0
                                   const bool conv);//true

int main()
{
    cudaError_t err;
    char filename_img[1024] = "image.test.data";//"image.one.data";
    char filename_filter[1024] = "filter.test.data";//"filter.zero.data";
    char filename_targetInit[1024] = "targetInit.test.data";

    float *images_h = readMatrix(filename_img, 9216, 1);
    float *images_d = NULL;
    float *filters_h = readMatrix(filename_filter, 1600, 64);
    float *filters_d = NULL;

    err = cudaMalloc((void**)&images_d, 9216*1*sizeof(float));
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");
    err = cudaMemcpy(images_d, images_h, (size_t)(9216*1*sizeof(float)), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");
    err = cudaMalloc((void**)&filters_d , 1600*64*sizeof(float));
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");
    err = cudaMemcpy(filters_d, filters_h, (size_t)(1600*64*sizeof(float)), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");

    float* targets_h = readMatrix(filename_targetInit, 9216, 1);
    float* targets_d = NULL;
    err = cudaMalloc((void**)&targets_d, 9216*1*sizeof(float));
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");
    err = cudaMemcpy(targets_d, targets_h, (size_t)(9216*1*sizeof(float)), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");

    //~~~~~~~~~~~~~~~~~~~~~~~~~~
        int imgSizeY = 12;
        int numModulesY = 12;
        int numModulesX = 12;
        int paddingStart = -2;
        int moduleStride = 1;
        int numImgColors = 64;
        int numGroups = 1;
        float scaleTargets = 0.0;
        float scaleOutput = 1.0;
        bool conv = true;

        int numFilterColors = numImgColors / numGroups;      
        int numFilters = 64;
        int numModules = numModulesY * numModulesX;
        int numImages = 1;
        int imgPixels = 9216/numImgColors;
        int imgSizeX = imgPixels / imgSizeY;
        int filterModuleMult = conv ? 1 : numModules;
        
        assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 2 == 0)));
        assert(numGroups == 1 || numFilterColors % 2 == 0);
        assert(numFilters % (16 * numGroups) == 0);
        assert(numImgColors % numGroups == 0);
        assert(9216 == imgPixels * numImgColors);
        assert(imgSizeY * imgSizeX == imgPixels);
        int numFiltersPerGroup = numFilters / numGroups;

        int imgStride = 1; // images does not need to be a contiguous matrix

        int filterPixels = 1600 / (filterModuleMult * numFilterColors);
        int filterSize = int(sqrt(filterPixels));
        assert(filterSize * filterSize == filterPixels);
        assert(1600 == filterModuleMult * numFilterColors * filterPixels);

        // These routines don't handle the case when only part of the image is visited in the convolution
        assert(paddingStart <= 0);
        assert(paddingStart + (numModulesX-1)*moduleStride + filterSize >= imgSizeX);
        assert(paddingStart + (numModulesY-1)*moduleStride + filterSize >= imgSizeY);
        assert(moduleStride <= filterSize);

        int imgsPerThread = numImages % 128 == 0 ? 4 : numImages % 64 == 0 ? 2 : 1;
        bool checkImgBounds = numImages % (32*imgsPerThread) != 0;
        if (scaleTargets == 0) {
            ;//targets.resize(numFilters * numModules, numImages);
        } else {
            assert(9216 == numFilters * numModules);
            assert(1 == numImages);
        }
        assert(imgSizeY == 12 && imgSizeX == 12 && numModulesY == 12 && numModulesX == 12);
        assert(filterSize == 5 && paddingStart == -2);
        assert(numFilters == 64);
        assert(numImgColors == 64);
    	assert(numGroups == 1);//sure
    	assert(scaleTargets == 0.0);//sure
    	assert(scaleOutput == 1.0);//sure
    	assert(numFiltersPerGroup == 64);//sure
    	assert(imgsPerThread == 1 && checkImgBounds == true);
    	if(!(numImages == 1 && imgStride == 1)) EXIT_MSG("ERROR !! this should no mini version(i.e. --mini=1)");
    //~~~~~~~~~~~~~~~~~~~~~~~~~~

    dim3 blocks (64, 1, 1);
    dim3 threads(144, 1, 1);
    
    //cudaFuncCachePreferNone//cudaFuncCachePreferShared//cudaFuncCachePreferL1
	cudaFuncSetCacheConfig(optiConvFilter <1>, cudaFuncCachePreferShared);
	optiConvFilter <1><<<blocks, threads>>>(images_d, filters_d, targets_d,
	    numImages, numFilters, imgSizeY, imgSizeX, filterSize, paddingStart, moduleStride, numModulesY,
	    numModulesX, imgStride, numImgColors, scaleTargets, scaleOutput, conv);

    err = cudaMemcpy(targets_h, targets_d, (size_t)(9216*1*sizeof(float)), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) EXIT_MSG("ERROR ~");
    print_result(targets_h, 9216, 1,9216, 1, 1);

}

template <int T_numImages>
__global__ void optiConvFilter(float* images, float* filters, float* targets,
                                   const int numImages, const int numFilters, //128(1), 64
                                   const int imgSizeY, const int imgSizeX, //12, 12
                                   const int filterSize, const int paddingStart, //5, -2
                                   const int moduleStride, //1
                                   const int numModulesY, const int numModulesX, //12, 12
                                   const int imgStride, const int numImgColors, //128(1), 64
                                   const float scaleTargets, const float scaleOutputs, //0.0, 1.0
                                   const bool conv)//true
{
	const int nMaxConnPerNeuron = (filterSize*filterSize) * numImgColors;
    const int neuronIdx = blockIdx.x;//blockDim.x*blockIdx.x + threadIdx.x;//0-63
    const int tarIdx = blockDim.x*(neuronIdx) + threadIdx.x;//blockDim.y*blockIdx.y + threadIdx.y;//0-9216
    const int imgPixels = imgSizeX*imgSizeY; //144
    const unsigned nLoads = nMaxConnPerNeuron;
    float privProd;
    float a=0.0;
    __shared__ float shW[1600];
    const unsigned loc = neuronIdx;//for first weight in that neuron

    privProd = 0.0;

    
    for (int i = 0; i < (nMaxConnPerNeuron-1)/imgPixels + 1; ++i){
        if(blockDim.x*i + threadIdx.x < nMaxConnPerNeuron)
            shW[blockDim.x * i + threadIdx.x] = filters[ loc + numFilters*(blockDim.x * i + threadIdx.x) ];
    }
    __syncthreads();

    int center = tarIdx % imgPixels;//img center : neuronIdx w/o color info
    int upperLeft = center - ((filterSize)/2) - imgSizeX*((filterSize)/2);
    //check padding condition
    //   * 1 *
    //   2   3
    //   * 4 *
    int padding1 = 0;
    int padding2 = 0;
    int padding3 = 0;
    int padding4 = 0;
    for (int i = 0; i < filterSize/2; ++i){
        for(int j = 0; j < i+1 ; j ++){
            padding1 += (int)(center/imgSizeX == j);
            padding2 += (int)(center%imgSizeX == j);
            padding3 += (int)(((imgSizeX - 1) - center%imgSizeX) == j);
            padding4 += (int)(((imgSizeX - 1) - center/imgSizeX) == j);
        }
    }
    //~~~~~~~~~~~~~~~iterate for 1(T_numImages) img~~~~~~~~~~~
	for (int li = 0; li < nLoads; li+= 1){  //0-1599
		int actLoadIdx = li;
		int c = actLoadIdx/(filterSize*filterSize); //color
		int y = (actLoadIdx%(filterSize*filterSize))/filterSize; // y idx in 5x5 filter
		int x = (actLoadIdx%(filterSize*filterSize))%filterSize; // x idx in 5x5 filter
        //w = filters[ loc + numFilters*actLoadIdx ];
        a = 0.0;
		if(y >= padding1 && (filterSize - 1) - y >= padding4 ){
			if(x >= padding2 && (filterSize - 1) - x >= padding3 ){
                a = images[(c*(imgPixels) + upperLeft + y*imgSizeX + x)*numImages + 0];
			}
		}
        privProd += a * shW[actLoadIdx];

	}
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /*
    * Store Phase
    */
    targets[tarIdx * numImages] = privProd;
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
























