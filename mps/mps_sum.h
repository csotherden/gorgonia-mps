// mps_sum.h
// Minimal C interface for invoking a Metal-based row-wise summation
// from Go via cgo. Each call uses the engine-level MPSEngineContext.
//
// This computes, for a row-major [rows x cols] matrix X:
//   y[i] = sum_j X[i, j]
// producing a length-rows output vector y.

#pragma once

#include "mps_engine_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

// mpsRowSumFloat32 performs a row-wise sum over a row-major [rows x cols]
// float32 matrix X, writing the per-row sums into the output vector y
// (length = rows) using the given engine context.
//
// Returns 0 on success, non-zero on failure. On failure, callers should
// fall back to a CPU implementation.
int mpsRowSumFloat32(MPSEngineContext ctx,
                     const float *x,
                     float *y,
                     int rows,
                     int cols);

#ifdef __cplusplus
}
#endif

