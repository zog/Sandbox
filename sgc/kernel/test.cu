#include <cutil_inline.h>
//#include <shrUtils.h>
#include <cuda.h>
#include <math.h>
//#include <mongo.h>

#define DIMENSIONS 5
#define BLOCK_SIZE 16 


// Kernel definition 
//__global__ void MatAdd(float A[N][N], float B[N][N], 
//                       float C[N][N]) 
//{ 
//    int i = threadIdx.x; 
//    int j = threadIdx.y; 
//    C[i][j] = A[i][j] + B[i][j];
//}

// Matrices are stored in row-major order: 
// M(row, col) = *(M.elements + row * M.width + col) 

__global__ void MatPopulate(float *A, int count) 
{ 
    int row = blockIdx.x;
    int col = threadIdx.x;
    A[row * DIMENSIONS + col] = (float)(row * DIMENSIONS + col)/(DIMENSIONS*count);
} 
 
float score(float *A, float *B){
  float score = 0.0;
  for(int i=0; i<DIMENSIONS; i++){
    //printf("%f * %f\n", A[i], B[i]);
    score += pow(A[i] * B[i], 2);
  }
  return score;
}

__global__ void ParallelScore(float *A, float *scores, int count) 
{ 
    float *q = &A[blockIdx.x*BLOCK_SIZE + threadIdx.x];
    float *p;
    __syncthreads();
    float score;
    score = 0.0;
    for(int j=0; j<count*DIMENSIONS; j+=DIMENSIONS){
      p = &A[j];
      __syncthreads();
      for(int i=0; i<DIMENSIONS; i++){
        score += pow(q[i] * p[i], 2);
      }
      //scores[blockIdx.x * COUNT + threadIdx.x] = score;
      __syncthreads();
    }
    scores[threadIdx.x] = score;
} 
 
int main(int argc, char *argv[])
{ 
    int count = (int)atoi(argv[1]) * 16;
    
    size_t size = DIMENSIONS * count * sizeof(float); 
    size_t size2 = BLOCK_SIZE * sizeof(float); 
    float *elements = (float *)malloc(size);
    float *d_elements;
    float *scores = (float *)malloc(size2);
    float *d_scores;
    cudaMalloc(&d_elements, size);
    cudaMalloc(&d_scores, size2);
    int threadsPerBlock = DIMENSIONS; 
    int numBlocks = count;
    MatPopulate<<<numBlocks, threadsPerBlock>>>(d_elements, count);
    cudaMemcpy(elements, d_elements, size, 
               cudaMemcpyDeviceToHost);
    for(int i=0; i<BLOCK_SIZE; i++){
      scores[i] = 0.0;
    }
    cudaMemcpy(d_scores, scores, size2, 
               cudaMemcpyHostToDevice);
    
    //for(int i=0; i<COUNT*DIMENSIONS; i+=DIMENSIONS){
    //  for(int j=0; j<DIMENSIONS; j++){
    //    printf("%f ", elements[i + j]);
    //  }
    //  printf("\n");
    //}
    
    if(argc > 2 && !strcmp(argv[2], "raw")){
      printf("\nraw\n");
      float _score;
      for(int i=0; i<count*DIMENSIONS; i+=DIMENSIONS){
        float *q = &elements[i];
        _score = 0.0;
        for(int j=0; j<count*DIMENSIONS; j+=DIMENSIONS){
          _score += score(q, &elements[j]);
        }
        scores[i % BLOCK_SIZE] = _score;
      }
    }
    else{
      printf("\nparallel\n");
      
      threadsPerBlock = BLOCK_SIZE; 
      numBlocks = count / threadsPerBlock;
      printf("\n%i blocks\n\n", numBlocks);
      ParallelScore<<<numBlocks, threadsPerBlock>>>(d_elements, d_scores, count);
      cudaMemcpy(scores, d_scores, size2, 
               cudaMemcpyDeviceToHost);
    }
    float sum = 0.0;
    for (int i=0;i<BLOCK_SIZE;i++)
    {
      sum += scores[i];
      printf("%f\n", scores[i]);
    }
    
    printf("%f\n", sum);
    cudaFree(d_elements);
    cudaFree(d_scores);
    free(elements);
    return EXIT_SUCCESS;
}