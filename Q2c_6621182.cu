/*************************************
**Question 2c
*************************************/

#include <stdio.h>

/*
 * Refactor `loop` to be a CUDA Kernel. The new kernel should
 * only do the work of 1 iteration of the original loop.
 */

 // Added a global function definition so the function can be ran
 // by the kernel
__global__ void loop(int N)
{
  for (int i = 0; i < N; ++i)
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
   * For this question, only use 1 block of threads.
  */

  /*
   * We are defining the size of blocks here and the number of threads,
   * This question requires that the kernel only use 1 block of threads
   * and only 1 block is required as this will only need to run 1 time.
   *
   * The number of threads in a block will be defined by the variable
   * blockSize.
   *
   * The number of blocks will be defined by the variable numBlocks.
   *
  */
	int numBlocks = 1;
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
