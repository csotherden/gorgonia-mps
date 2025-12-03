package mps

import (
	"math"
	"math/rand"
	"testing"

	"gorgonia.org/tensor"
)

// newRandomFloat32MatrixForSum creates a 2D float32 dense tensor with
// deterministic pseudo-random contents for Sum tests.
func newRandomFloat32MatrixForSum(t *testing.T, rows, cols int, r *rand.Rand) *tensor.Dense {
	t.Helper()
	size := rows * cols
	data := make([]float32, size)
	for i := range data {
		data[i] = float32(r.NormFloat64())
	}
	return tensor.New(
		tensor.WithShape(rows, cols),
		tensor.WithBacking(data),
	)
}

// equalApproxF32 reports whether two float32 slices are equal within a tolerance.
func equalApproxF32(a, b []float32, tol float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		diff := float32(math.Abs(float64(a[i] - b[i])))
		if diff > tol {
			return false
		}
	}
	return true
}

// Test that for the supported case (2D float32, axis last dim), MPSEng.Sum
// matches StdEng.Sum within a small numerical tolerance.
func TestMPSEngSumLastAxisMatchesStdEng(t *testing.T) {
	r := rand.New(rand.NewSource(99))

	const (
		rows = 8
		cols = 16
	)

	x := newRandomFloat32MatrixForSum(t, rows, cols, r)

	var cpu tensor.StdEng
	cpuOut, err := cpu.Sum(x, 1)
	if err != nil {
		t.Fatalf("StdEng.Sum error: %v", err)
	}

	mpsEng := NewMPSEng()
	mpsOut, err := mpsEng.Sum(x, 1)
	if err != nil {
		t.Fatalf("MPSEng.Sum error: %v", err)
	}

	cpuDense, ok := cpuOut.(*tensor.Dense)
	if !ok {
		t.Fatalf("StdEng.Sum did not return *tensor.Dense, got %T", cpuOut)
	}
	mpsDense, ok := mpsOut.(*tensor.Dense)
	if !ok {
		t.Fatalf("MPSEng.Sum did not return *tensor.Dense, got %T", mpsOut)
	}

	logicalLen := cpuDense.Shape().TotalSize()
	got := cpuDense.Data().([]float32)[:logicalLen]
	want := mpsDense.Data().([]float32)[:logicalLen]

	if !equalApproxF32(got, want, 1e-4) {
		t.Fatalf("MPSEng.Sum result differs from StdEng.Sum.\n got:  %v\n want: %v", got, want)
	}
}

// --- Benchmarks ------------------------------------------------------------

// benchmarkSum is a helper that benchmarks either StdEng.Sum or MPSEng.Sum
// on a single (rows x cols) matrix, summing over the last axis.
func benchmarkSum(b *testing.B, rows, cols int, useMPS bool) {
	b.Helper()

	r := rand.New(rand.NewSource(42))

	// Base tensor with default (StdEng) engine.
	xCPU := newRandomFloat32MatrixForSum(&testing.T{}, rows, cols, r)

	var (
		cpu    tensor.StdEng
		mpsEng = NewMPSEng()
		x      tensor.Tensor
	)

	if useMPS {
		// Reuse the same backing but attach MPSEng as the engine so that
		// the internal MatMul in MPSEng.Sum will use the GPU-backed path.
		backing, ok := xCPU.Data().([]float32)
		if !ok {
			b.Fatalf("expected []float32 backing, got %T", xCPU.Data())
		}
		x = tensor.New(
			tensor.WithShape(rows, cols),
			tensor.WithBacking(backing),
			tensor.WithEngine(mpsEng),
		)
	} else {
		x = xCPU
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if useMPS {
			if _, err := mpsEng.Sum(x, 1); err != nil {
				b.Fatalf("MPSEng.Sum error: %v", err)
			}
		} else {
			if _, err := cpu.Sum(x, 1); err != nil {
				b.Fatalf("StdEng.Sum error: %v", err)
			}
		}
	}
}

func BenchmarkStdEngSum_128x128(b *testing.B) {
	benchmarkSum(b, 128, 128, false)
}

func BenchmarkMPSEngSum_128x128(b *testing.B) {
	benchmarkSum(b, 128, 128, true)
}

func BenchmarkStdEngSum_512x512(b *testing.B) {
	benchmarkSum(b, 512, 512, false)
}

func BenchmarkMPSEngSum_512x512(b *testing.B) {
	benchmarkSum(b, 512, 512, true)
}

func BenchmarkStdEngSum_1024x1024(b *testing.B) {
	benchmarkSum(b, 1024, 1024, false)
}

func BenchmarkMPSEngSum_1024x1024(b *testing.B) {
	benchmarkSum(b, 1024, 1024, true)
}
