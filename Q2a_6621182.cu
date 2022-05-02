/*************************************
**Question 2a
*************************************/
#include <stdio.h>




void helloCPU(){
    printf("Hello from the CPU.\n");
}

// Added a global function defintion so that it can be run the device.
__global__ void helloGPU()
{
	// Altered the string from Hello from the CPU.\n to Hello from the GPU.\n
	// This is to tell the user that this (GPU) function is being shown.
	printf("Hello also from the GPU.\n");
}

// Write a function that outputs to the console from the GPU

// Main function definition where there are no parameters.
int main(void)
{
	// Only using 1 block as only required to do this once.
	int numBlocks = 1;
	// Only using block size of 1 as only require to do this once.
	int blockSize = 1;

	// Because we are running this on the GPU have to specify the number
	// of blocks and the block size.
	helloGPU<<<numBlocks,blockSize>>>();

	// Synchronise the GPU
	cudaDeviceSynchronize();

	// Run the normal CPU function after the GPU has synchronised.
	helloCPU();

	// The return type of the function is int so returned 0 as the
	// program has exited successfully.
	return 0;

}

