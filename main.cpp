#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <vector>
#include <immintrin.h>

#define FMT_HEADER_ONLY

#include "fmt/core.h"

/*
 * Author: Ahmed Yehia
 * Date: 2021 November 6
 * Student ID: 220125711
 * Instructor: Dr. Mohamed Jeragh
 * Course: CpE 561
 * Assignment 1
 * The purpose of this assignment is to parallelize a normal for-loop using threading and SIMD operations.
 * There are three approaches highlighted and profiled here: threading, SIMD, and threading + SIMD.
 * lscpu Output:
 * Architecture:                    x86_64
 * CPU op-mode(s):                  32-bit, 64-bit
 * Byte Order:                      Little Endian
 * Address sizes:                   39 bits physical, 48 bits virtual
 * CPU(s):                          8
 * On-line CPU(s) list:             0-7
 * Thread(s) per core:              2
 * Core(s) per socket:              4
 * Socket(s):                       1
 * NUMA node(s):                    1
 * Vendor ID:                       GenuineIntel
 * CPU family:                      6
 * Model:                           142
 * Model name:                      Intel(R) Core(TM) i5-10210U CPU @ 1.60GHz
 * Stepping:                        12
 * CPU MHz:                         2100.000
 * CPU max MHz:                     4200.0000
 * CPU min MHz:                     400.0000
 * BogoMIPS:                        4199.88
 * Virtualization:                  VT-x
 * L1d cache:                       128 KiB
 * L1i cache:                       128 KiB
 * L2 cache:                        1 MiB
 * L3 cache:                        6 MiB
 * NUMA node0 CPU(s):               0-7
 * Vulnerability Itlb multihit:     KVM: Mitigation: VMX disabled
 * Vulnerability L1tf:              Not affected
 * Vulnerability Mds:               Not affected
 * Vulnerability Meltdown:          Not affected
 * Vulnerability Spec store bypass: Mitigation; Speculative Store Bypass disabled 
 *                                  via prctl and seccomp
 * Vulnerability Spectre v1:        Mitigation; usercopy/swapgs barriers and __use
 *                                  r pointer sanitization
 * Vulnerability Spectre v2:        Mitigation; Enhanced IBRS, IBPB conditional, R
 *                                  SB filling
 * Vulnerability Srbds:             Mitigation; TSX disabled
 * Vulnerability Tsx async abort:   Not affected
 * Flags:                           fpu vme de pse tsc msr pae mce cx8 apic sep mt
 *                                  rr pge mca cmov pat pse36 clflush dts acpi mmx
 *                                   fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb
 *                                   rdtscp lm constant_tsc art arch_perfmon pebs 
 *                                  bts rep_good nopl xtopology nonstop_tsc cpuid 
 *                                  aperfmperf pni pclmulqdq dtes64 monitor ds_cpl
 *                                   vmx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pci
 *                                  d sse4_1 sse4_2 x2apic movbe popcnt tsc_deadli
 *                                  ne_timer aes xsave avx f16c rdrand lahf_lm abm
 *                                   3dnowprefetch cpuid_fault epb invpcid_single 
 *                                  ssbd ibrs ibpb stibp ibrs_enhanced tpr_shadow 
 *                                  vnmi flexpriority ept vpid ept_ad fsgsbase tsc
 *                                  _adjust bmi1 avx2 smep bmi2 erms invpcid mpx r
 *                                  dseed adx smap clflushopt intel_pt xsaveopt xs
 *                                  avec xgetbv1 xsaves dtherm ida arat pln pts hw
 *                                  p hwp_notify hwp_act_window hwp_epp md_clear f
 *                                  lush_l1d arch_capabilities
 */

void simpleImage(short row, short col, int *r, int *g, int *b) {

    // Simple for-loop to generate the image
    for (int i = 0; i < row * col; i++) {
        r[i] = 256 * i / col / row;
        g[i] = 256 * (i % col) / col;
        b[i] = 128;
    }
}

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

long timeIt(const std::string &functionName, void (*func)(short, short, int *, int *, int *)) {

    /* I've moved the timing from the entire function to just the logic, since the biggest bottleneck is the speed
     * of writing to the
     */
    std::cout << "================================================================================\n";
    std::cout << fmt::format("Profiling function: {}\n", functionName);
    const unsigned int row = 1080;
    const unsigned int col = 1920;
    int numberOfSamples = 10;
    std::chrono::nanoseconds avg = std::chrono::nanoseconds::zero();
    std::chrono::nanoseconds min = std::chrono::nanoseconds::max();
    std::chrono::nanoseconds max = std::chrono::nanoseconds::min();

    for (int i = 0; i < numberOfSamples; i++) {
        // Each color gets its own array for memory access performance
        auto *r = new int[row * col];
        auto *g = new int[row * col];
        auto *b = new int[row * col];
        std::cout << fmt::format("Loop iteration: {}\n", i + 1);
        // Reasonably big buffer in order to write the file to disk only once
        std::vector<char> buffer(33554432);
        std::fstream out;
        out.rdbuf()->pubsetbuf(&buffer.front(), static_cast<long>(buffer.size()));
        out.open(fmt::format("../{}.ppm", functionName), std::ios::out);
        out << fmt::format("P3\n{} {}\n255\n", col, row);
        // time the performance of the approach itself, not the writing to disk
        auto t1 = std::chrono::high_resolution_clock::now();
        func(row, col, r, g, b);
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < row * col; j++) {
            out << fmt::format("{} {} {}\n", r[j], g[j], b[j]);
        }
        out.close();
        std::chrono::nanoseconds diff = t2 - t1;
        std::cout << fmt::format("Loop execution time: {} ns\n\n", diff.count());
        avg += diff;
        max = max < diff ? diff : max;
        min = min > diff ? diff : min;
    }

    std::cout << fmt::format("Minimum execution time: {} ns\n", min.count());
    std::cout << fmt::format("Average execution time: {} ns\n", avg.count() / numberOfSamples);
    std::cout << fmt::format("Maximum execution time: {} ns\n", max.count());
    std::cout << "================================================================================\n";
    // Use the minimum time for measurement of speedup
    return min.count();
}

int main() {
    auto spTime = static_cast<double>(timeIt("simpleImage", simpleImage));
    auto thTime = static_cast<double>(timeIt("threadedImage", threadedImage));
    auto sdTime = static_cast<double>(timeIt("simdImage", simdImage));
    auto approxSdTime = static_cast<double>(timeIt("approximatedSimdImage", approximatedSimdImage));
    auto thSdTime = static_cast<double>(timeIt("threadedSimdImage", threadedSimdImage));

    std::cout << "================================================================================\n";
    std::cout << fmt::format("Threaded measured speedup over simple: {0:.1f}%\n", spTime / thTime * 100);
    std::cout << fmt::format("SIMD measured speedup over simple: {0:.1f}%\n", spTime / sdTime * 100);
    std::cout << fmt::format("Approximated SIMD measured speedup over simple: {0:.1f}%\n", spTime / approxSdTime * 100);
    std::cout << fmt::format("Threaded + SIMD measured speedup over simple: {0:.1f}%\n", spTime / thSdTime * 100);
    std::cout << "================================================================================\n";

    return 0;
}
