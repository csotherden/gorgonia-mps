// mps_matmul.h
// Minimal C interface for invoking Metal Performance Shaders matrix
// multiplication from Go via cgo. Each MPSEng holds its own opaque
// engine context instead of using global Metal state.

#pragma once

#include "mps_engine_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

// mpsMatMulFloat32 performs C = A x B using Metal Performance Shaders
// with the given context.
//
// A is an m x k row-major matrix, B is a k x n row-major matrix, and
// C is an m x n row-major matrix. All matrices are stored in contiguous
// float32 buffers.
//
// Returns 0 on success, non-zero on failure. On failure, callers should
// fall back to a CPU implementation.
int mpsMatMulFloat32(MPSEngineContext ctx,
                     const float *a,
                     const float *b,
                     float *c,
                     int m,
                     int n,
                     int k);

#ifdef __cplusplus
}
#endif


