/*************************************
**Question 2d
*************************************/

#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

__global__ void loop(int N)
{
	/*
	 * Because the function is launched on a 'grid' which is divided into 'blocks'
	 * and each block is divided into threads, we want each instance to access a different
	 * portion of the buffer.
	 *
	 * To calculate which thread we are currently working on we want to
	 * use the following index and stride to make sure that the numbers
	 * are only outputted once
	*/

	int index = blockIdx.x * blockDim.x + threadIdx.x;

	int stride = blockDim.x * gridDim.x;

	for (int i = index; i < N; i+= stride)
	{
		printf("This is iteration number %d\n", i);
	}
}

int main(void)
{
  /*
   * When refactoring `loop` to launch as a kernel, be sure
   * to use the execution configuration to control how many
   * "iterations" to perform.
   *
   * For this question, use at least 2 blocks of threads.
   */

  /*
   * We are defining the size of blocks here and the number of threads,
   * This question requires that the kernel uses 2 blocks of threads
   *
   * The number of threads in a block will be defined by the variable
   * blockSize.
   *
   * The number of blocks will be defined by the variable numBlocks.
   *
  */
	int numBlocks = 2;
	int blockSize = 1;

	// N is the number of iterations that will be outputted.
	int N = 10;

	// Calling the function to be ran by the kernel using 1 block and 1 thread.
	loop<<<numBlocks, blockSize>>>(N);

	// Synchronize the device
	cudaDeviceSynchronize();

	// The return type of the function is int so returned 0 as the
	// program has exited successfully.
	return 0;
}
