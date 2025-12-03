// mps_engine.go
package mps

import "gorgonia.org/tensor"

// MPSEng is a tensor.Engine implementation that currently just delegates
// everything to tensor.StdEng. The MatMul method is explicitly wrapped so
// you can later replace its body with a concrete MPS/Metal implementation.
type MPSEng struct {
	tensor.StdEng
}

// NewMPSEng constructs a new MPSEng.
//
// For now there’s nothing to initialize beyond the embedded StdEng, but this
// gives you a single place to add MPS/Metal setup later (device, queue, etc.).
func NewMPSEng() *MPSEng {
	return &MPSEng{
		StdEng: tensor.StdEng{},
	}
}

// Compile-time check that *MPSEng satisfies tensor.Engine.
var _ tensor.Engine = (*MPSEng)(nil)
