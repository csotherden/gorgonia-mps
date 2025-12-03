//go:build !darwin || !cgo

// mps_matmul.go (CPU fallback)
package mps

import "gorgonia.org/tensor"

// MatMul is a thin wrapper around the CPU StdEng.MatMul implementation.
//
// This is the method you’ll replace with your MPS-backed matmul once the
// wiring is verified and you have your Metal/MPS plumbing in place.
func (e *MPSEng) MatMul(a, b, prealloc tensor.Tensor) error {
	return e.StdEng.MatMul(a, b, prealloc)
}
