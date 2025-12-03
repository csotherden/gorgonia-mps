//go:build darwin && cgo

// mps_matmul_darwin.go
//
// Darwin-only MatMul implementation that offloads 2D float32 matrix
// multiplication to Metal Performance Shaders (MPS) while keeping
// tensor allocations in regular Go/CPU memory.

package mps

/*
#cgo darwin CFLAGS: -fobjc-arc
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "mps_matmul.h"
*/
import "C"

import (
	"fmt"

	"gorgonia.org/tensor"
)

// isRowMajorContiguous2D reports whether d is a 2D dense tensor with the
// standard row-major layout that our simple MPS wrapper expects:
//
//	shape = [rows, cols]
//	strides = [cols, 1]
func isRowMajorContiguous2D(d *tensor.Dense) bool {
	if d.Dims() != 2 {
		return false
	}
	shape := d.Shape()
	strides := d.Strides()
	if len(shape) != 2 || len(strides) != 2 {
		return false
	}
	rows, cols := shape[0], shape[1]
	return strides[1] == 1 && strides[0] == cols && rows > 0 && cols > 0
}

// MatMul offloads 2D float32 matrix multiplication (with standard
// row‑major layout) to Metal Performance Shaders. For everything else
// (non-dense tensors, non-float32 dtypes, non-2D shapes, non-standard
// strides, or any MPS failure) it transparently falls back to the
// embedded StdEng implementation.
func (e *MPSEng) MatMul(a, b, prealloc tensor.Tensor) error {
	// Fast path only for dense, float32, 2D row‑major matrices.
	da, okA := a.(*tensor.Dense)
	db, okB := b.(*tensor.Dense)
	dc, okC := prealloc.(*tensor.Dense)
	if !okA || !okB || !okC {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	if da.Dtype() != tensor.Float32 || db.Dtype() != tensor.Float32 || dc.Dtype() != tensor.Float32 {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	if !isRowMajorContiguous2D(da) || !isRowMajorContiguous2D(db) || !isRowMajorContiguous2D(dc) {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	shapeA := da.Shape()
	shapeB := db.Shape()
	shapeC := dc.Shape()

	m, kA := shapeA[0], shapeA[1]
	kB, n := shapeB[0], shapeB[1]

	if kA != kB {
		return fmt.Errorf("mps: MatMul shape mismatch: a=%v, b=%v (inner dims %d vs %d)", shapeA, shapeB, kA, kB)
	}
	if len(shapeC) != 2 || shapeC[0] != m || shapeC[1] != n {
		return fmt.Errorf("mps: MatMul prealloc shape mismatch: expected [%d %d], got %v", m, n, shapeC)
	}

	adata, ok := da.Data().([]float32)
	if !ok {
		return e.StdEng.MatMul(a, b, prealloc)
	}
	bdata, ok := db.Data().([]float32)
	if !ok {
		return e.StdEng.MatMul(a, b, prealloc)
	}
	cdata, ok := dc.Data().([]float32)
	if !ok {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	// Basic length sanity checks in case we're dealing with views.
	if len(adata) < m*kA || len(bdata) < kB*n || len(cdata) < m*n {
		return fmt.Errorf("mps: MatMul backing slice too small: a=%d, b=%d, c=%d, expected at least %d, %d, %d",
			len(adata), len(bdata), len(cdata), m*kA, kB*n, m*n)
	}

	status := C.mpsMatMulFloat32(
		(*C.float)(&adata[0]),
		(*C.float)(&bdata[0]),
		(*C.float)(&cdata[0]),
		C.int(m),
		C.int(n),
		C.int(kA),
	)

	// On any MPS error, fall back to the CPU implementation so that
	// callers still get correct results even on non-Metal systems or
	// if something goes wrong in the GPU path.
	if status != 0 {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	return nil
}
