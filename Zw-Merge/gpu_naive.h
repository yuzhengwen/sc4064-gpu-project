#pragma once

/*
 * gpu_naive.h — Baseline GPU merge sort (global memory only)
 *
 * Bottom-up iterative merge sort.  One thread per merge pair; all data lives in global memory throughout
 *
 * Returns total kernel time in ms (excludes H2D / D2H transfers)
 */

float run_gpu_naive(const int *h_src, int *h_dst, int n);
