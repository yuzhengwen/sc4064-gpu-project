#pragma once

/*
 * Allocates device memory, copies h_src → device, sorts with thrust::sort, copies result → h_dst.
 * Returns pure sort kernel time in ms (H2D and D2H excluded)
 */

float run_thrust_sort(const int *h_src, int *h_dst, int n);
