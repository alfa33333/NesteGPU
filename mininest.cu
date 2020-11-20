// starting snippet for nested sampling
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <float.h>

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d -- %s\n",__FILE__,__LINE__, cudaGetErrorString(x)); \
return EXIT_FAILURE;}} while(0)
    
#define N 32
    
__global__ void setup_kernel ( curandState * state, unsigned long seed )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init ( seed, idx, 0, &state[idx] );
} 
    
__global__ void generate( curandState* globalState, float * randomArray ) 
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = globalState[idx];
    float RANDOM = curand_uniform( &localState );
    randomArray[idx] = RANDOM;
    globalState[idx] = localState;
}


int main( int argc, char** argv) 
{
    
    dim3 threads = dim3(N, 1);
    int blocksCount = 1;// floor(N / threads.x) + 1;
    dim3 blocks  = dim3(blocksCount, 1);
    curandState* devStates;
    float * randomValues = new float[N];
    float * devRandomValues;
    
    printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n",
    blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);

    CUDA_CALL(cudaMalloc ( &devStates, N*sizeof( curandState ) ));
    CUDA_CALL(cudaMalloc ( &devRandomValues, N*sizeof( *randomValues ) ));
    
    // setup seeds
    setup_kernel <<<blocks, threads>>> ( devStates, time(NULL) );

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    // generate random numbers
    generate <<<blocks, threads>>> ( devStates, devRandomValues );

    printf("%s\n", cudaGetErrorString(cudaGetLastError()));

    CUDA_CALL(cudaMemcpy      ( randomValues, devRandomValues, N*sizeof(*randomValues), cudaMemcpyDeviceToHost ));
    
    for(int i=0;i<N;i++)
    {
        printf("#%i %f\n",i, randomValues[i]);
    }
    
    
    CUDA_CALL(cudaFree(devRandomValues));
    CUDA_CALL(cudaFree(devStates));
    
    delete randomValues;

    return 0;
}