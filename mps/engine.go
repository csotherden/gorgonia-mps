// mps_engine.go
package mps

import (
	"unsafe"

	"gorgonia.org/tensor"
)

// MPSEng is a tensor.Engine implementation that embeds tensor.StdEng but
// also holds an opaque handle to a GPU context used by the MPS-backed
// operations (matmul and future ops).
type MPSEng struct {
	tensor.StdEng
	ctx unsafe.Pointer
}

// NewMPSEng constructs a new MPSEng.
//
// For now there’s nothing to initialize beyond the embedded StdEng, but this
// gives you a single place to add MPS/Metal setup later (device, queue, etc.).
func NewMPSEng() *MPSEng {
	e := &MPSEng{
		StdEng: tensor.StdEng{},
	}
	initMPSEngine(e)
	return e
}

// Compile-time check that *MPSEng satisfies tensor.Engine.
var _ tensor.Engine = (*MPSEng)(nil)
