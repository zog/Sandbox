extern "C" {
  #include "kernel.h"

  __global__ void MatPopulate(float *A, int count) 
  { 
    int row = blockIdx.x;
    int col = threadIdx.x;
    A[row * DIMENSIONS + col] = (float)(row * DIMENSIONS + col)/(DIMENSIONS*count);
    //A[row * DIMENSIONS + col] = (float)(row + col)/(BLOCK_SIZE);
  } 
  
  float score(float *A, float *B){
    float score = 0.0;
    for(int i=0; i<DIMENSIONS; i++){
      //printf("%f * %f\n", A[i], B[i]);
      score += A[i] * B[i];
    }
    return score;
  }
  
  __global__ void ParallelScore(float *A, float *B, int* keysA, int* keysB, float *scores, int count)
  { 
    float *q = &A[(blockIdx.x*blockDim.x + threadIdx.x)*DIMENSIONS];
    int *keys_q = &keysA[(blockIdx.x*blockDim.x + threadIdx.x)*DIMENSIONS];
    
    float *p;
    int *keys_p;
    __syncthreads();
    float _score;
    for(int j=0; j<count; j++){
      _score = 0.0;
      p = &B[j*DIMENSIONS];
      keys_p = &keysB[j*DIMENSIONS];
      
      __syncthreads();
      for(int k=0; k<DIMENSIONS; k++){
        int key = keys_q[k];
        for(int l=0; l<DIMENSIONS; l++){
          if(keys_p[l] == key){
            _score += q[k] * p[l];
          }
        }
        
      }
      __syncthreads();
      scores[(blockIdx.x*blockDim.x + threadIdx.x) * BLOCK_SIZE + j] = (blockIdx.x*blockDim.x + threadIdx.x) * BLOCK_SIZE + j;//sqrt(_score);
    }
  } 
}
