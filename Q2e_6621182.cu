/*************************************
**Question 2e
*************************************/
#include <stdio.h>

/**
 * Initialises the list of values of array a to be
 * the increasing values from 0 to N.
 *
 * e.g N = 10: [0,1,2,3,4,5,6,7,8,9]
 */
void init(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    a[i] = i;
  }
}

/**
 * Function ran on the device that calculates the current thread
 * and given the thread is less than the number of elements in the
 * list 'a' the value at the thread index of the list 'a' is doubled.
 */
__global__ void doubleElements(int *a, int N)
{
  int i;
  i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
  {
    a[i] *= 2;
  }
}

/**
 * Function that returns whether the elements in the list
 * have been doubled.
 *
 * If true then boolean value of true will be returned.
 *
 * Not not all been doubled then false will be returned.
 */
bool checkElementsAreDoubled(int *a, int N)
{
  int i;
  for (i = 0; i < N; ++i)
  {
    if (a[i] != i*2) {
    	return false;
    }
  }
  return true;
}

int main()
{
  // Set the number of elements in the input array to be 100.
  int N = 100;

  // Create a pointer to a integer array 'a'.
  int *a;

  // Calculate the size of the integer list
  size_t size = N * sizeof(int);

  // Refactored the previous memory allocation (a = (int *)malloc(size)) to
  // provide a pointer 'a' that can be used on both the host and the device.

  // Allocate the size number of bytes of the nunber of elements in the input
  // array multiplied by the number of bytes of the integer type
  // so that we can store N integer elements in the array 'a'
  cudaMallocManaged(&a, size);

  // Initialises the values of the array 'a' using the function init
  init(a, N);

  // Setting the nunber of threads per block to be 10.
  size_t threads_per_block = 10;

  // Setting the number of blocks being ran 10.
  size_t number_of_blocks = 10;

  // Invoke the doubleElements kernel to run 100 times.
  doubleElements<<<number_of_blocks, threads_per_block>>>(a, N);

  // Synchronize the device code.
  cudaDeviceSynchronize();

  // Check that the elements of the array 'a' are all doubled.
  bool areDoubled = checkElementsAreDoubled(a, N);

  // Output whether the elements of the array 'a' are doubled.
  printf("All elements were doubled? %s\n", areDoubled ? "TRUE" : "FALSE");

  // Refactored (free(a)) to free the memory allocated by the cudaMallocManaged
  // that can be accessed by both the host and the device.
  cudaFree(a);
}
