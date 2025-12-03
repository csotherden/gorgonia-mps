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
#include "mps_engine_ctx.h"
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

// denseToRowMajor2DF32 materializes the logical contents of a 2D float32
// Dense tensor into a row-major contiguous []float32 buffer.
//
// If the tensor is already row-major contiguous and doesn't require an
// iterator, the returned slice is just the underlying backing slice and
// alias=true. Otherwise a fresh buffer is allocated, the values are copied
// in logical (row, col) order, and alias=false.
func denseToRowMajor2DF32(d *tensor.Dense) (buf []float32, alias bool, err error) {
	if d.Dtype() != tensor.Float32 {
		return nil, false, fmt.Errorf("denseToRowMajor2DF32: expected Float32, got %v", d.Dtype())
	}
	if d.Dims() != 2 {
		return nil, false, fmt.Errorf("denseToRowMajor2DF32: expected 2D tensor, got %dD", d.Dims())
	}

	shape := d.Shape()
	rows, cols := shape[0], shape[1]
	if rows == 0 || cols == 0 {
		return nil, false, fmt.Errorf("denseToRowMajor2DF32: zero-sized matrix %v", shape)
	}

	data, ok := d.Data().([]float32)
	if !ok {
		return nil, false, fmt.Errorf("denseToRowMajor2DF32: backing is %T, want []float32", d.Data())
	}

	// Fast path: already row-major contiguous and no iterator needed.
	if !d.RequiresIterator() && isRowMajorContiguous2D(d) {
		need := rows * cols
		if len(data) < need {
			return nil, false, fmt.Errorf("denseToRowMajor2DF32: backing slice too small: have %d, need %d", len(data), need)
		}
		return data[:need], true, nil
	}

	// General path: use the tensor's iterator to respect its logical layout
	// (including slices, transposes, masks, etc.) and write into a compact
	// row-major buffer.
	buf = make([]float32, rows*cols)

	it := d.Iterator()
	for idx, e := it.Start(); !it.Done(); idx, e = it.Next() {
		if e != nil {
			return nil, false, fmt.Errorf("denseToRowMajor2DF32: iterator error: %w", e)
		}
		coord := it.Coord()
		if len(coord) != 2 {
			return nil, false, fmt.Errorf("denseToRowMajor2DF32: expected 2D coord, got %dD", len(coord))
		}
		r, c := coord[0], coord[1]
		if r < 0 || r >= rows || c < 0 || c >= cols {
			return nil, false, fmt.Errorf("denseToRowMajor2DF32: coord out of range: %v for shape %v", coord, shape)
		}
		buf[r*cols+c] = data[idx]
	}

	return buf, false, nil
}

// rowMajor2DToDenseF32 writes the contents of a row-major contiguous
// buffer back into a 2D float32 Dense tensor. If the tensor is already
// row-major contiguous and doesn't require an iterator, this is a single
// copy. Otherwise it scatters into the tensor using its iterator to
// respect arbitrary view layouts.
func rowMajor2DToDenseF32(buf []float32, d *tensor.Dense) error {
	if d.Dtype() != tensor.Float32 {
		return fmt.Errorf("rowMajor2DToDenseF32: expected Float32, got %v", d.Dtype())
	}
	if d.Dims() != 2 {
		return fmt.Errorf("rowMajor2DToDenseF32: expected 2D tensor, got %dD", d.Dims())
	}

	shape := d.Shape()
	rows, cols := shape[0], shape[1]
	if rows == 0 || cols == 0 {
		return fmt.Errorf("rowMajor2DToDenseF32: zero-sized matrix %v", shape)
	}
	if len(buf) < rows*cols {
		return fmt.Errorf("rowMajor2DToDenseF32: buf too small: have %d, need %d", len(buf), rows*cols)
	}

	data, ok := d.Data().([]float32)
	if !ok {
		return fmt.Errorf("rowMajor2DToDenseF32: backing is %T, want []float32", d.Data())
	}

	// Fast path: direct copy into backing slice.
	if !d.RequiresIterator() && isRowMajorContiguous2D(d) {
		copy(data[:rows*cols], buf)
		return nil
	}

	// General path: scatter from row-major buffer into the tensor's layout
	// using its iterator.
	it := d.Iterator()
	for idx, e := it.Start(); !it.Done(); idx, e = it.Next() {
		if e != nil {
			return fmt.Errorf("rowMajor2DToDenseF32: iterator error: %w", e)
		}
		coord := it.Coord()
		if len(coord) != 2 {
			return fmt.Errorf("rowMajor2DToDenseF32: expected 2D coord, got %dD", len(coord))
		}
		r, c := coord[0], coord[1]
		if r < 0 || r >= rows || c < 0 || c >= cols {
			return fmt.Errorf("rowMajor2DToDenseF32: coord out of range: %v for shape %v", coord, shape)
		}
		data[idx] = buf[r*cols+c]
	}

	return nil
}

// MatMul offloads 2D float32 matrix multiplication (with standard
// row‑major layout) to Metal Performance Shaders when possible. For all
// other supported 2D float32 layouts (transposed views, sliced views,
// non-row-major but contiguous, etc.) it materializes temporary
// row‑major buffers before/after the GPU call. For non-dense tensors,
// non-float32 dtypes, non-2D shapes, or any MPS failure it transparently
// falls back to the embedded StdEng implementation.
func (e *MPSEng) MatMul(a, b, prealloc tensor.Tensor) error {
	// Fast path only for dense, float32, 2D matrices; layout is handled
	// internally via denseToRowMajor2DF32/rowMajor2DToDenseF32.
	da, okA := a.(*tensor.Dense)
	db, okB := b.(*tensor.Dense)
	dc, okC := prealloc.(*tensor.Dense)
	if !okA || !okB || !okC {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	if da.Dtype() != tensor.Float32 || db.Dtype() != tensor.Float32 || dc.Dtype() != tensor.Float32 {
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

	// Materialize A and B as row-major 2D float32 buffers. For base
	// row‑major tensors this is just a view on the underlying backing
	// slice; for transposed/sliced views this allocates a temporary
	// buffer and copies via iterator.
	abuf, _, err := denseToRowMajor2DF32(da)
	if err != nil {
		return e.StdEng.MatMul(a, b, prealloc)
	}
	bbuf, _, err := denseToRowMajor2DF32(db)
	if err != nil {
		return e.StdEng.MatMul(a, b, prealloc)
	}

	// Decide how to handle the output: if the prealloc tensor is already
	// row‑major contiguous with a simple backing slice, let MPS write
	// into it directly. Otherwise, write into a temporary row‑major
	// buffer and scatter back into the tensor afterwards.
	var (
		cbuf      []float32
		useDirect bool
	)

	cdata, ok := dc.Data().([]float32)
	if ok && !dc.RequiresIterator() && isRowMajorContiguous2D(dc) && len(cdata) >= m*n {
		cbuf = cdata[:m*n]
		useDirect = true
	} else {
		cbuf = make([]float32, m*n)
		useDirect = false
	}

	status := C.mpsMatMulFloat32(
		(C.MPSEngineContext)(e.ctx),
		(*C.float)(&abuf[0]),
		(*C.float)(&bbuf[0]),
		(*C.float)(&cbuf[0]),
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

	// If we wrote into a temporary buffer, scatter back into the logical
	// layout of the prealloc tensor.
	if !useDirect {
		if err := rowMajor2DToDenseF32(cbuf, dc); err != nil {
			// As a safety net, fall back to CPU on any unexpected layout
			// issue during scatter.
			return e.StdEng.MatMul(a, b, prealloc)
		}
	}

	return nil
}
