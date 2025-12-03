//go:build !darwin || !cgo

// engine_other.go
//
// Non-darwin (or non-cgo) stub for MPSEng initialization. On these
// platforms, MPSEng simply delegates to tensor.StdEng without any
// additional setup.

package mps

func initMPSEngine(e *MPSEng) {
	_ = e
	// No-op on non-Metal platforms.
}


