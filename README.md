# CpE 561 Assignment 1
This repository contains the code that covers the requirements of the first assignment of the CpE 561 Parallel Computing course at KU in the Fall 2021 semester.

### Introduction

One of the fundamental structures of any programming language is for loops. For loops are used extensively for any
operation that requires multiple iterations over the same piece of code. The problem with loops is that they can take
a long time to execute due to their serial nature (one iteration at a time) and due to how complex the code contained
may be. Not only that, but modern CPU architecture includes parallelism in many ways, notably by threading as well as
SIMD (single instruction multiple data) operations. Simple for loops do not take advantage of these structures and in
fact end up wasting a lot of the CPU's potential. The architecture of CPUs has been improved over time to optimise loop
execution, but more speedup could be gained simply by taking advantage of already-existing features in hardware.

The code written in the assignment covers the following approaches:
1) Simple for-loop approach
2) Threaded for-loop approach
3) SIMD for-loop approach
4) Approximated SIMD for-loop approach
5) Threaded + SIMD for-loop approach

It's important to note that the profiling of all these approaches share a few things in common:
- The measurements are taken for the logic of the code itself, not IO operations, since the IO operations are the
bottleneck for all approaches.
- The arrays that hold the color values is not initialised within any of the approaches; indeed, any static initialisation
that could be done outside the approaches and is common to all of them was extracted.
- The minimum execution time for each approach was used for comparison against the simple approach; this is to ensure that
each approach's maximum potential is explored.
- There are three arrays of length `row * col` each holding a color.
- The image is represented in code as a 1D array and all approaches deal with it as such.

### 1) Simple for-loop approach

In the simple approach, the program loops over each of the three arrays, calculates the value, and updates the array elements
accordingly. The following bit of code highlights this approach:

```c++
void simpleImage(short row, short col, int *r, int *g, int *b) {

    // Simple for-loop to generate the image
    for (int i = 0; i < row * col; i++) {
        r[i] = 256 * i / col / row;
        g[i] = 256 * (i % col) / col;
        b[i] = 128;
    }
```

Using this method on an Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz processor produced the following minimum time in a
random sample: 29025627 ns

And this is the image produced by the simple method.

![Simple approach](simdImage.bmp)

### 2) Threaded for-loop approach

In the threaded approach, the program first queries the system for available hardware threads, then divides the work 
into equal chunks based on the result (assuming the work is equally divisible by the number of threads available).
The biggest challenge faced was finding a way to let the threads do their work and then join it in the correct order;
the arrays' structure as well as careful planning allows them all to finish, await each other via `thread.join`, then 
return to allow the timing function to print the result to output. Since the task is trivially parallelized,
the division of work was relatively simple. The following bit of code highlights this approach:

```c++
void threadedImage(short row, short col, int *r, int *g, int *b) {

    // Here the work is split into threads; each thread takes a piece of the for-loop and works on it
    unsigned short numberOfThreads = std::thread::hardware_concurrency();
    numberOfThreads = numberOfThreads == 0 ? 2 : numberOfThreads;
    auto *threads = new std::thread[numberOfThreads];

    unsigned int start = 0;
    unsigned int end = row * col / numberOfThreads;
    for (unsigned int count = 0; count < numberOfThreads; count++) {
        threads[count] = std::thread([col, row, r, g, b](int start, int end) {
            for (int i = start; i < end; i++) {
                r[i] = 256 * i / col / row;
                g[i] = (256 * (i % col)) / col;
                b[i] = 128;
            }
        }, start, end);
        start += row * col / numberOfThreads;
        end += row * col / numberOfThreads;
    }

    // Wait for all the threads to finish executing before returning
    for (unsigned short count = 0; count < numberOfThreads; count++) {
        threads[count].join();
    }
}
```

Using this method on an Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz processor produced the following minimum time in a
random sample: 7109078 ns

And this is the image produced by the threaded method.

![Threaded approach](threadedImage.bmp)

### 3) SIMD for-loop approach

In the SIMD approach, the array values were worked on 8 values at a time. The challenge was figuring out how to utilise
vector instructions in the best possible way. To this end, the colors were separated into three separate arrays, and
each array's values were directly loaded and stored using the `_mm256_loadu_epi32` and `_mm256_storeu_epi32` instructions
and providing the pointer to the 8 32-bit integer block that the operation was intended for.
The following bit of code highlights this approach:

```c++
void simdImage(short row, short col, int *r, int *g, int *b) {

    const int addVal[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    /*
     * Here the work is done 8 pixels at a time, in order to take advantage of SIMD instructions.
     * The results are directly loaded into the respective arrays 256 bits at a time.
     */
    for (int i = 0; i < row * col / 8; i++) {
        _mm256_storeu_si256((__m256i *) &r[i * 8],
                            _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(
                                                                      _mm256_add_epi32(
                                                                              _mm256_set1_epi32(i * 8),
                                                                              _mm256_loadu_si256((__m256i *) addVal))),
                                                              _mm256_set1_ps(256.0f / static_cast<float>(col) /
                                                                             static_cast<float>(row)))));
        _mm256_storeu_si256((__m256i *) &g[i * 8],
                            _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(
                                                                      _mm256_add_epi32(
                                                                              _mm256_set1_epi32((i * 8 % col)),
                                                                              _mm256_loadu_si256((__m256i *) addVal))),
                                                              _mm256_set1_ps(256.0f / static_cast<float>(col)))));
        _mm256_storeu_si256((__m256i *) &b[i * 8], _mm256_set1_epi32(128));
    }
}
```

Using this method on an Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz processor produced the following minimum time in a
random sample: 10667309 ns

And this is the image produced by the SIMD method.

![SIMD approach](simdImage.bmp)

### 4) Approximated SIMD for-loop approach

In the approximated SIMD approach, it was noted that the operations could be slowed down due to the conversion from
integer to float, performing a multiplication, then returning to integer. Thus, an approximated that lumps each 8 pixels
in a row together in order to reduce computational cost was presented. In this approach, the speedup gained was not noteworthy,
but it was measurable, suggesting that there could be other use cases that would indeed favor approximations over
accurate calculations. The following bit of code highlights this approach:

```c++
void approximatedSimdImage(short row, short col, int *r, int *g, int *b) {

    /*
     * Here the work is done 8 pixels at a time, in order to take advantage of SIMD instructions.
     * The results are directly loaded into the respective arrays 256 bits at a time.
     */

    for (int i = 0; i < row * col / 8; i++) {
        _mm256_storeu_si256((__m256i *) &r[i * 8], _mm256_set1_epi32(256 * (8 * i + 3) / (row * col)));
        _mm256_storeu_si256((__m256i *) &g[i * 8], _mm256_set1_epi32(256 * ((8 * i + 3) % col) / col));
        _mm256_storeu_si256((__m256i *) &b[i * 8], _mm256_set1_epi32(128));
    }
}
```

Using this method on an Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz processor produced the following minimum time in a
random sample: 9391629 ns

And this is the image produced by the approximated SIMD method.

![Threaded approach](approximatedSimdImage.bmp)

### 5) Threaded + SIMD for-loop approach

In the threaded + SIMD approach, the objective was to utilize the full power that each core offers. By threading the work,
each physical core available was used, and by utilizing SIMD operations, each core's vector ALUs were put to work. This
method yielded almost a 10x increase in performance without compromising image quality.
Implementing it required careful combination of the threaded approach and the SIMD approach.
The following bit of code highlights this approach:

```c++
void threadedSimdImage(short row, short col, int *r, int *g, int *b) {

    // This approach combines threading and SIMD to achieve up to x10 speedup on the arch that I developed the code on
    unsigned short numberOfThreads = std::thread::hardware_concurrency();
    numberOfThreads = numberOfThreads == 0 ? 2 : numberOfThreads;
    auto *threads = new std::thread[numberOfThreads];
    const int addVal[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    unsigned int start = 0;
    unsigned int end = row * col / numberOfThreads / 8;
    for (unsigned int count = 0; count < numberOfThreads; count++) {
        threads[count] = std::thread([col, row, r, g, b, addVal](int start, int end) {
            for (int i = start; i < end; i++) {
                _mm256_storeu_si256((__m256i *) &r[i * 8],
                                    _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(
                                                                              _mm256_add_epi32(
                                                                                      _mm256_set1_epi32(i * 8),
                                                                                      _mm256_loadu_si256((__m256i *) addVal))),
                                                                      _mm256_set1_ps(256.0f / static_cast<float>(col) /
                                                                                     static_cast<float>(row)))));
                _mm256_storeu_si256((__m256i *) &g[i * 8],
                                    _mm256_cvttps_epi32(_mm256_mul_ps(_mm256_cvtepi32_ps(
                                                                              _mm256_add_epi32(
                                                                                      _mm256_set1_epi32((i * 8 % col)),
                                                                                      _mm256_loadu_si256((__m256i *) addVal))),
                                                                      _mm256_set1_ps(
                                                                              256.0f / static_cast<float>(col)))));
                _mm256_storeu_si256((__m256i *) &b[i * 8], _mm256_set1_epi32(128));
            }
        }, start, end);
        start += row * col / numberOfThreads / 8;
        end += row * col / numberOfThreads / 8;
    }

    for (unsigned short count = 0; count < numberOfThreads; count++) {
        threads[count].join();
    }
}
```

Using this method on an Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz processor produced the following minimum time in a
random sample: 2982875 ns

And this is the image produced by the threaded + SIMD method.

![Threaded + SIMD approach](threadedSimdImage.bmp)

These are the final speedups achieved by each method:

```bash
================================================================================
Threaded measured speedup over simple: 408.3%
SIMD measured speedup over simple: 277.7%
Approximated SIMD measured speedup over simple: 329.6%
Threaded + SIMD measured speedup over simple: 994.3%
================================================================================
```