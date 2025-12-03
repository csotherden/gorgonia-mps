//go:build darwin && cgo

// engine_darwin.go
//
// Darwin-specific initialization for MPSEng. This ensures that the
// underlying Metal device and command queue used by the MPS-backed
// operations (currently MatMul, and future ops) are created eagerly
// when a new engine is constructed, rather than lazily inside each
// operation.

package mps

/*
#cgo darwin CFLAGS: -fobjc-arc
#cgo darwin LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation
#include "mps_engine_ctx.h"
*/
import "C"

import "unsafe"

// initMPSEngine performs any one-time Metal/MPS initialization needed
// for this engine instance. At the moment it simply creates a dedicated
// engine context; future GPU-backed ops can be wired through here as
// well.
func initMPSEngine(e *MPSEng) {
	e.ctx = unsafe.Pointer(C.MPSEngineCreateContext())
}
