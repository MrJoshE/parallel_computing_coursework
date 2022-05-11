#### **(a) What is the “>>” operator? [4] Explain its working in terms of powers of 2. Show its working, with all steps, for the case 12>>2.**

The ‘>>’ operator represents the bitwise right-shift operator, which shifts the bits to the right.

Right shifting an integer **x** with an integer **y** is the same as dividing $x$ by $2^y$

The operation `12>>2` would be as follows: 
$$
\frac{12}{2^2} =
\frac{12}{4}=
3
$$

#### **(b) Ignoring overheads due to memory access and task communication, calculate the pessimistic and optimistic estimates of overall speedup (up to 2 decimal places) for the case where 75% of the work in a task is sped up by the use of 1000 parallel processors. [4]**

We have $f = 0.25$ and $p=1000$,  so using this formula 
$$
\psi = \frac{1}{f + \frac{1 - f}{p}}
$$
We can get the optimistic estimate of overall speedup

$$
y=\frac{1}{0.25 + \frac{1 - 0.25}{1000}} = 3.988
$$
Then we can calulate the parallel overhead using the following function to get the parallel overhead time.
$$
\begin{align*}
e&= \frac{\frac{1}{\psi}-\frac{1}{p}}{1-\frac{1}{p}}\\
&= \frac{\frac{1}{3.988}-\frac{1}{1000}}{1-\frac{1}{1000}}\\
&= 0.2501
\end{align*}
$$
Now we just add the parallel overhead to the optimistic time $3.988 - 0.2501 = 3.7379$

And we have:

- Optimistic speedup: 3.988
- Pessimistic speedup: 3.7379

#### (c) Assuming that that a kernel called someKernel has been defined, how many times will it run based on the following invocation: someKernel<<<10, 10>>>() [3]

someKernel<<<10, 10>>() is configured to run in 10 thread blocks which each have 10 threads and will therefore run 100 times.

#### (d) What is the arithmetic intensity of the matrix multiplication kernel using shared memory (Lab class week 3)? [5]

$$
\frac{A.width multiplications + A.width additions}{2(A.width) reads from shared memory + A.width global memory access}
$$

####  (e) Assume that we want to use each thread to calculate two (adjacent) elements of a vector addition. What would be the expression for mapping the thread/block indices to i, the data index of the first element to be processed by a thread? [4] 

###### I. i=blockIdx.x*blockDim.x + threadIdx.x +2; 

###### II. i=blockIdx.x*threadIdx.x*2 

###### III. i=(blockIdx.x\*blockDim.x + threadIdx.x)*2 

###### IV. i=blockIdx.x\*blockDim.x*2 + threadIdx.x 

Answer: III

#### (f) If a CUDA device’s SM (streaming multiprocessor) can take up to 1536 threads and up to 4 thread blocks. Which of the following block configuration would result in the most number of threads in the SM? [4] 

###### I. 128 threads per block 

###### II. 256 threads per block 

###### III. 512 threads per block 

###### IV. 1024 threads per block 

Answer: III

#### (g) In order to write a kernel that operates on an image of size 400x900 pixels, you would like to assign one thread to each pixel. You would like your thread blocks to be square and to use the maximum number of threads per block possible on the device (your device has maximum number of threads per block as 1024). How would you select the grid dimensions and block dimensions of your kernel? [6]

Grid -> Block -> Thread

First I would define the size of a block, and taking into account the limit size of a block (1024) I would choose the size of `blockSize = 32 x 32`, as that is $\sqrt{1024}=32$. Since the image is 2D, I would us a 2D block and grid shape to process it. Once I have defined the block size I would proceed  to choose the grid dimension. When calculating the grid dimension, I would make sure to round up the number to make sure there are enoughs threads. Thus the 2D grid dimension would be $400/32=13$ and $900/32=29$, which would amount to a total of 416 x 928 threads.

So we have that `2DblockSize = 32 x 32` and `2DgridSize = 13 x 29`. As there are more threads than pixels, we would have to implement some kind of bound checks that would only allow threads that are within the image bounds to do processing, as otherwise it might cause some errors because the threads would be accessing memory outside the image bounds.

