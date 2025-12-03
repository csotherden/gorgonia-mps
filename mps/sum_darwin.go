//go:build darwin && cgo

// sum_darwin.go
//
// Darwin-only Sum override for MPSEng that uses a dedicated Metal
// Performance Shaders reduction kernel (via a small C bridge) to
// accelerate common 2D float32 reductions used in attention (summing
// over the last dimension). For all other cases it falls back to the
// embedded StdEng implementation.

package mps

/*
#cgo darwin CFLAGS: -fobjc-arc
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "mps_sum.h"
*/
import "C"

import "gorgonia.org/tensor"

// resolveAxis mirrors tensor.resolveAxis (which is unexported) so that
// we can support negative axes in a consistent way for Sum.
//
// For example, for dims=2 and axis=-1 this returns 1 (the last dim).
func resolveAxis(axis, dims int) int {
	res := axis % dims
	if (res < 0 && dims > 0) || (res > 0 && dims < 0) {
		return res + dims
	}
	return res
}

// Sum accelerates the pattern:
//   - a is *tensor.Dense with dtype Float32
//   - a has rank 2
//   - along has exactly one axis, which is the last dimension (axis=-1 or axis=Dims()-1)
//
// In that case, it computes the sum over the last dimension via a
// dedicated MPS reduction kernel. For all other inputs it defers to
// StdEng.Sum.
func (e *MPSEng) Sum(a tensor.Tensor, along ...int) (tensor.Tensor, error) {
	// Only handle the simple 2D single-axis case here; everything else
	// goes through the default StdEng implementation.
	if len(along) != 1 {
		return e.StdEng.Sum(a, along...)
	}

	ad, ok := a.(*tensor.Dense)
	if !ok {
		return e.StdEng.Sum(a, along...)
	}
	if ad.Dtype() != tensor.Float32 {
		return e.StdEng.Sum(a, along...)
	}

	if ad.Dims() != 2 {
		return e.StdEng.Sum(a, along...)
	}

	axis := resolveAxis(along[0], ad.Dims())
	if axis != ad.Dims()-1 {
		// For now we only accelerate sum over the last dimension.
		return e.StdEng.Sum(a, axis)
	}

	shape := ad.Shape()
	rows, cols := shape[0], shape[1]
	if rows == 0 || cols == 0 {
		return e.StdEng.Sum(a, axis)
	}

	// For now we only support straightforward row-major, non-iterator
	// layouts. More complex views fall back to the CPU implementation.
	if ad.RequiresIterator() {
		return e.StdEng.Sum(a, axis)
	}

	data, ok := ad.Data().([]float32)
	if !ok {
		return e.StdEng.Sum(a, axis)
	}
	if len(data) < rows*cols {
		return e.StdEng.Sum(a, axis)
	}

	status := C.mpsRowSumFloat32(
		(C.MPSEngineContext)(e.ctx),
		(*C.float)(&data[0]),
		(*C.float)(&data[0]), // in-place: y overwrites the first rows entries
		C.int(rows),
		C.int(cols),
	)

	if status != 0 {
		// GPU path failed – fall back to CPU.
		return e.StdEng.Sum(a, axis)
	}

	// The first 'rows' elements of the backing slice now contain the
	// per-row sums. Reshape to a 1D vector of length rows.
	if err := ad.Reshape(rows); err != nil {
		return ad, nil
	}

	return ad, nil
}
