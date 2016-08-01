#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <math.h>

using namespace std;

#define datafloat float

#define BDIM 256

__global__ void partialSum(const int N,
			   datafloat *u,
			   datafloat *blocksum){

  __shared__ datafloat s_blocksum[BDIM];

  int t = threadIdx.x;  
  int b = blockIdx.x;
  const int n = b*blockDim.x + t;
  
  s_blocksum[t] = 0;
  
  // prefetch one entry per thread to shared memory
  if(n < N){
    s_blocksum[t] = u[n];
  }

  // initially tag all threads as alive
  int alive = blockDim.x;

  while(alive>1){

    __syncthreads();  // barrier (make sure s_red is ready)
    
    alive /= 2;
    if(t < alive)
      s_blocksum[t] += s_blocksum[t+alive];
  }
  
  // value in s_blocksum[0] is sum of block of values
  if(t==0) 
    blocksum[b] = s_blocksum[0];
}
  

// same partial sum reduction, but with unrolled while loop
__global__ void unrolledPartialSum(const int entries,
				   datafloat *u,
				   datafloat *blocksum){

  __shared__ datafloat s_red[BDIM];

  int t = threadIdx.x;  
  int b = blockIdx.x;
  const int n = b*blockDim.x + t;

  s_red[t] = 0;
  
  if(n<N){
    s_red[t] = u[n];
  }

  __syncthreads();  // barrier (make sure s_red is ready)

  // manually unrolled reduction (assumes BDIM=256)
  if(BDIM>128) {
    if(t<128)
      s_red[t] += s_red[t+128];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>64){
    if(t<64)
      s_red[t] += s_red[t+64];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>32){
    if(t<32)
      s_red[t] += s_red[t+32];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>16){
    if(t<16)
      s_red[t] += s_red[t+16];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>8){
    if(t<8)
      s_red[t] += s_red[t+8];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>4){
    if(t<4)
      s_red[t] += s_red[t+4];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>2){
    if(t<2)
      s_red[t] += s_red[t+2];

    __syncthreads();  // barrier (make sure s_red is ready)
  }

  if(BDIM>1){
    if(t<1)
      s_red[t] += s_red[t+1];
  }

  // store result of this block reduction
  if(t==0)
    red[b] = s_red[t];
}

void reduction(int N, datafloat tol, datafloat *h_u){

  // Device Arrays
  datafloat *c_u, *c_partialsum;

  // Host array for partial sum
  datafloat *h_partialsum;

  // number of thread-blocks to partial sum u
  int B = (N+BDIM-1)/BDIM;

  // allocate host array
  h_partialsum = (datafloat*) calloc(B, sizeof(datafloat));

  // allocate device arrays
  cudaMalloc((void**) &c_u  , N*sizeof(datafloat));
  cudaMalloc((void**) &c_partialsum , B*sizeof(datafloat));

  // copy from h_u to c_u (HOST to DEVICE)
  cudaMemcpy(c_u ,  h_u ,  N*sizeof(datafloat), cudaMemcpyHostToDevice);
  
  // Create CUDA events
  cudaEvent_t startEvent, endEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&endEvent);

  cudaEventRecord(startEvent, 0);

  // perform reduction 10 times
  int Ntests = 10, test;
  datafloat res;

  for(test=0;test<Ntests;++test){
    reduction <<< bdim, gdim >>> (N, c_u, c_partialsum);

    // Finish reduce on host
    cudaMemcpy(h_partialsum, c_partialsum, B*sizeof(datafloat), cudaMemcpyDeviceToHost);

    datafloat psum = 0;
    for(int n=0;n<B;++n){
      psum += h_partialsum[n];
    }
    printf("sum total = %g\n", psum);

  }
  
  cudaEventRecord(endEvent, 0);
  cudaEventSynchronize(endEvent);

  // Get time taken
  float timeTaken;
  cudaEventElapsedTime(&timeTaken, startEvent, endEvent);

  const datafloat avgTimePerTest = timeTaken/((datafloat) Ntests);

  // free device arrays
  cudaFree(c_u);
  cudaFree(c_partialsum);

  // free HOST array
  free(h_partialsum);

}

int main(int argc, char** argv){

  // parse command line arguements
  if(argc != 2){
    printf("Usage: ./main N \n");
    return 0;
  }

  // Number of internal domain nodes in each direction
  const int N     = atoi(argv[1]);

  // Host Arrays
  datafloat *h_u   = (datafloat*) calloc(N, sizeof(datafloat));
  
  // initialize host array
  for(int n = 0; i < N; ++i){
    h_u[n] = 1;
  }

  // Solve discrete Laplacian
  reduction(N, h_u);

  // Free the host array
  free(h_u);
}
