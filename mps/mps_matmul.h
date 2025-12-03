// mps_matmul.h
// Minimal C interface for invoking Metal Performance Shaders matrix
// multiplication from Go via cgo.

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

// mpsMatMulFloat32 performs C = A x B using Metal Performance Shaders.
//
// A is an m x k row-major matrix, B is a k x n row-major matrix, and
// C is an m x n row-major matrix. All matrices are stored in contiguous
// float32 buffers.
//
// Returns 0 on success, non-zero on failure. On failure, callers should
// fall back to a CPU implementation.
int mpsMatMulFloat32(const float *a,
                     const float *b,
                     float *c,
                     int m,
                     int n,
                     int k);

#ifdef __cplusplus
}
#endif


