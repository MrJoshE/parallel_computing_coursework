/*************************************
**Question 2b
*************************************/
#include <stdio.h>

__global__ void printSuccessForCorrectExecutionConfiguration()
{

  if(threadIdx.x == 1023 && blockIdx.x == 255)
  {
    printf("Success!\n");
  }
  else {
    printf("Failure. Update the execution configuration as necessary.\n");
  }
}

// Main function definition where there are no parameters.
int main(void)
{

	/*
	 * We are defining the size of blocks here and the number of threads,
	 * This question requires that the kernel will print success
	 * the only way that it will is if the thread id is 1023 and the
	 * block id is 255.
	 *
	 * The number of threads in a block will be defined by the variable
	 * blockSize.
	 *
	 * The number of blocks will be defined by the variable numBlocks.
	 *
	*/

	// Number of threads in a block is set to 1024.
	int blockSize = 1024;

	// Number of blocks is set to 256
	int numBlocks = 256;

	/*
	   * Update the execution configuration so that the kernel
	   * will print `"Success!"`.
	*/
	printSuccessForCorrectExecutionConfiguration<<<numBlocks,blockSize>>>();



  // Synchronize the device
  cudaDeviceSynchronize();

  // The return type of the function is int so returned 0 as the
  // program has exited successfully.
  return 0;
}
