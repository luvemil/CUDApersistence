// Compute the distance matrix (only in the upper diagonal)
__global__ void ComputeDistanceMatrix(float Data[N][M],float DistMatrix[M][M], int blockIncr)
{
  // Ci sono problemi di bordo
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int i,j;
  if(by > bx) {
    i = (bx + blockIncr) * blockDim.x + threadIdx.x;
    j = (by + blockIncr) * blockDim.y + threadIdx.y;
  } else if(bx > by) {
    i = bx * blockDim.x + threadIdx.x;
    j = by * blockDim.y + threadIdx.y;
  } else {
    if(threadIdx.y > threadIdx.x) {
      /*  If I am in the lower triangle, simply transpose the index (after moving
          to the lower part of the matrix)
      */
      j = (bx + blockIncr) * blockDim.x + threadIdx.x;
      i = (by + blockIncr) * blockDim.y + threadIdx.y;
    } else {
      i = bx * blockDim.x + threadIdx.x;
      j = by * blockDim.y + threadIdx.y;
    }
  }
  if (i==j) {
    DistMatrix[i][j] = 0;
    return;
  } else if (i > M || j > M){
    return;
  }

  // Compute the Euclidian distance between points i and j
  int span;
  float sqDist = 0;
  for(span = 0; span < N; span++) {
    float diff = Data[span][i] - Data[span][j];
    sqDist += diff * diff;
  }
  float dist = sqrtf(sqDist);
  DistMatrix[i][j] = dist;
}

int main()
{
  // Memory Allocation etc. goes here.

  //check rounding
  dim3 threadsPerBlock(16,16);
  dim3 numBlocks(M / threadsPerBlock.x, M / (2 * threadsPerBlock.y));
  ComputeDistanceMatrix<<<numBlocks, threadsPerBlock>>>(Data, DistMatrix,numBlocks.y);
}
