#include <stdio.h>

// Simple helper function that linearizes the id
__device__ int get_id(dim3 idx, dim3 dim){
  return
    idx.x +
    idx.y * dim.x +
    idx.z * dim.x * dim.y;
}

__global__ void whoami() {
  // Defines the id of the block inside the grid
  int block_id = get_id(blockIdx, gridDim);

  // Defines how many thread came before that block
  int block_offset = 
    block_id
    * blockDim.x * blockDim.y * blockDim.z;

  // Defines the id of the thread inside the block
  int thread_offset = get_id(threadIdx, blockDim);

  // Final id is the how many threads came before + the thread offset inside the current block
  int id = block_offset + thread_offset;

  printf("%04d | Block(%d %d %d) = %3d | Thread(%d %d %d) = %3d\n",
        id,
        blockIdx.x, blockIdx.y, blockIdx.z, block_id,
        threadIdx.x, threadIdx.y, threadIdx.z, thread_offset);
}

int main() { 
  dim3 blocksPerGrid(2, 2, 2);
  dim3 threadsPerBlock(2, 1, 1);

  int blocks_per_grid = blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;
  int threads_per_block = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z;
  printf("%d blocks/grid\n", blocks_per_grid);
  printf("%d threads/block\n", threads_per_block);
  printf("%d total threads\n", blocks_per_grid * threads_per_block);

  whoami<<<blocksPerGrid, threadsPerBlock>>>();
  cudaDeviceSynchronize();

  return 0;
}
