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
 *
 * Below there is a definition of the number of elements in the
 * input array. Just change that number to change the size of the
 * input array.
*/
#define InputArraySize 4096


/*
 * Definition for global varibles that can be changed.
 */

// Represents the number of threads in each block.
#define BLOCK_SIZE 8

static int readCount = 0;

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
	CudaSafeCall(cudaMallocManaged(&idata, InputArraySize*sizeof(int)));


	printf("Allocated memory %ld bytes to the array data.\n", InputArraySize*sizeof(int) );

	//Initialise the input data on the host
	//Making it easy to test the result
	for(int i=0; i < InputArraySize; i++)
	{
		idata[i] = i;
	}

	printf("Populated array data with values up from 0 to %d.\n", InputArraySize);

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

	// Dynamic grid size so that the grid size can be dependent depending on the size of the input array and the block size
	// makes it more efficient.
	int grid_size = ceil(static_cast< int >(InputArraySize)/BLOCK_SIZE);

    // Define the variable that will hold the sum that will be returned.
	int sum = 0;

	/*
	 *  Create a pointer to an integer array in the variable dev_out.
	 *
	 *  This array will be an array that will contain the ongoing sum
	 *  of the array for each section of the sum.
	 *
	 *  The contents of this array will summed by the CPU to get the final
	 *  reduction.
	 *
	 */
	int* dev_out;

	// Allocate memory to the dev_out array at the size of the grids.

	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaMallocManaged(&dev_out, sizeof(int)* grid_size));


	// Call the reduce kernel function to get the partial results of the sum.
	reduceKernel<<<grid_size, BLOCK_SIZE>>>(InputArraySize, arr, dev_out);

	//dev_out now holds the partial result

	// Synchronise the device
	// Wrapping this in a CudaSafeCall function to handle the error if one is thrown.
	CudaSafeCall(cudaDeviceSynchronize());

	// Performing the final reduction on the CPU as only 1 kernal call was allowed.
	for (int i = 0; i < grid_size; i++){
		// Adds the element of the array to the sum
		sum += dev_out[i];
	}

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


	}

	// Update the value of the array element the thread is working on
	// with the value of temp
	// only ran when the thread id is 0.
	if (tid == 0){
		odata[blockIdx.x] = shArr[0];
	}

	//Once we have iterated through the for-loop, we will be left with the reduced value,
	//which is in the one belonging to thread index 0, i.e. at data (input array) index 0.
}
