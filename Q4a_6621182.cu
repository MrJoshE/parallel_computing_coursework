#include <stdio.h>
/*
 * Question 4
 *
 * Reduction of a 1D array of elements into a single sum value. Where the
 * size of the input array will require the use of multiple blocks of threads.
 *
*/

/*
 * Provide a complete implementation of a parallel algorithm that
 * will Reduce a 1D array of elements into a single summary value,
 * where the size of the input array will require the use of multiple
 * blocks of threads. The reduction should give the sum of the list.
*/



/*
 * Definition for global varibles that can be changed.
 */

// Represents the size of the grids.
#define GRID_SIZE 32
// Represents the number of blocks.
#define NUM_BLOCKS 4
// Represents the number of threads in each block.
#define BLOCK_SIZE 64
// Represents the number of elements in the input array
#define N 32

/*
 * Function that will perform reduction on sections of the input array.
 *
 * sizeOfArray - the length of the array [idata]
 *
 * idata - a pointer to the integer array of numbers that are being reduced
 *
 * odata - a pointer to the integer array of sums of sections of the input array.
 *
 * The purpose of the reduceKernel function is to reduce a 1D array into a single
 * summation of the integer elements of the array.
 *
 * This can run on multiple blocks of threads that can be customised using the
 * global definitions above.
 *
 */
__global__ void reduceKernel(int sizeOfArray, int *idata, int *odata);

/*
 * Sum array function that is defined with __host__ to tell the compiler that the function should
 * be ran on the CPU.
 *
 * This function receives a pointer to an integer array.
 *
 * The purpose of this function is handle calling the reduceKernel function and to sum
 * each of the sections returned by the reduceKernel function.
 *
 * This function is responsible for handling the memory allocation for the different lists that it
 * makes
 */
__host__ int sumArray(int* arr );

/**
 * Definition of the error handling functions
 *
 */
#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

/**
 * Function accepts a cuda error, file and line
 *
 * This will check that the err property was not a success and if so will output the error
 * that was returned.
 *
 * This is intended to be used to handle errors thrown by functions such as cudaMalloc / cudaMallocManaged etc.
 */
inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if(cudaSuccess != err)
	{
		fprintf( stderr, "cudaSafeCall() failed at %s:%i : %s\n",
				file, line, cudaGetErrorString( err ) );
		exit( -1 );
	}
}



/*
 * The main function that will run when the function is compiled.
 *
 */
int main(void)
{

	// Create a pointer to an array of integers and store in variable idata (input data).
	int *idata;

	//Allocate Unified Memory -- accessible from CPU and GPU
	//nice and simple, but it does mean that the
	// input array is overwritten to provide the result

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMallocManaged(&idata, N*sizeof(int)));


	printf("Allocated memory %ld bytes to the array data.\n", N*sizeof(int) );

	//Initialise the input data on the host
	//Making it easy to test the result
	for(int i=0; i < N; i++)
	{
		idata[i] = i;
	}
	printf("Populated array data with values up from 0 to %d.\n", N);

	// Call the sumArray function passing in the idata array to reduce the array
	// and store the result in the variable sum.
	int sum = sumArray(idata);

	//Now output the result of the reduction.
	printf("The final reduction is: %d\n", sum); //the reduced sum is contained

	//in index 0 of the array
	printf("\n");

	//Free all allocated memory

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaFree(idata));

	// Return 0 to show that the program executed successfully.
	return 0;
}

__host__ int sumArray(int* arr )
{

	/*
	 *  Create a pointer to an integer array in the variable dev_arr.
	 *
	 *  This array will be a temporary array that will have the contents
	 *  of the input array (arr) copied into it.
	 *
	 */

	int* dev_arr;

	// Allocate the dev_arr as much memory as the input arr has (as it will need to hold the same contents).


	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMalloc((void**)&dev_arr, N * sizeof(int)));

    // Copy the contents of arr into the dev_arr

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMemcpy(dev_arr, arr, N * sizeof(int), cudaMemcpyHostToDevice));

    // Define the variable that will hold the sum that will be returned.
	int sum;

	/*
	 *  Create a pointer to an integer array in the variable dev_out.
	 *
	 *  This array will be an array that will contain the ongoing sum
	 *  of the array for each section of the sum. The final sum will be taken from this array.
	 *
	 */
	int* dev_out;

	// Allocate memory to the dev_out array at the size of the grids.

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMalloc((void**)&dev_out, sizeof(int)* GRID_SIZE));


	// Call the reduce kernel function to get the partial results of the sum.
	reduceKernel<<<GRID_SIZE, BLOCK_SIZE>>>(N, dev_arr, dev_out);

	// Check to see if an error was thrown during the kernel invocation
	cudaError err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
		exit( -1 );
	}

	//dev_out now holds the partial result

	// Reduce the partial results into a single result stored index 0 of the dev_out array.
	reduceKernel<<<1, BLOCK_SIZE>>>( GRID_SIZE,dev_out, dev_out);

	// Check to see if an error was thrown during the kernel invocation
	err = cudaGetLastError();
	if ( cudaSuccess != err )
	{
		fprintf( stderr, "cudaCheckError() failed at %s:%i : %s\n", __FILE__, __LINE__, cudaGetErrorString( err ) );
		exit( -1 );
	}


	//dev_out[0] now holds the final result

	// Synchronise the device

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaDeviceSynchronize());

	// Copy the first value of the dev_out into the sum variable. Done by copying the amount of memory that a single integer takes
	// from the start pointer of the dev_out array into sum.

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMemcpy(&sum, dev_out, sizeof(int), cudaMemcpyDeviceToHost));

	// Free up the memory allocated to the dev_arr array.

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaFree(dev_arr));

	// Free up the memory allocated to the dev_out array.

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaFree(dev_out));

	// Return the resultant sum to the caller (main function).
	return sum;
}

__global__ void reduceKernel(int sizeOfArray, int *idata, int *odata)
{

	// mapping the local thread index onto the index
	//of the array element we are going to be working on.
	int tid = threadIdx.x;

	// mapping the local grid index that is being worked on.
	int gid = tid + blockIdx.x * BLOCK_SIZE;

	// Defining the grid size
	const int gridSize = BLOCK_SIZE * gridDim.x;

	// temporary sum of the section.
	int sum = 0;

	// For each of the grids sum its section of the input array
	for (int i = gid; i < sizeOfArray; i += gridSize) {
		sum += idata[i];
	}

	// Create a shared integer array with the size of the block
	__shared__ int shArr[BLOCK_SIZE];
	shArr[tid] = sum;

	// Sync the threads of the block
	__syncthreads();

	//do reduction
	// define the stride ‘stride’, start off with s equal to half the block size,
	// with each step through the for-loop, s is reduced to half of its previous value
	//by using the right-shift (>>) operator
	for( int stride=BLOCK_SIZE/2; stride>=1; stride>>=1){
		if(tid < stride){
			//get the value of the array element the thread is working on &
			//add to it the value of its neighbour ‘s’ elements away. Store it
			// in the shared array at the index of the current thread variable to avoid race condition
			 shArr[tid] += shArr[tid + stride];

		}

		// After the read, do barrier synchronisation to synch threads
		// to make sure all adds at one stage are done
		__syncthreads();

		// Update the value of the array element the thread is working on
		// with the value of temp
		if (tid == 0){
			odata[blockIdx.x] = shArr[0];
		}
	}
	//Once we have iterated through the for-loop, we will be left with the reduced value,
	//which is in the one belonging to thread index 0, i.e. at data (input array) index 0.
}
