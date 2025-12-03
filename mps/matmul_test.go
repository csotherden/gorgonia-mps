package mps

import (
	"math"
	"math/rand"
	"testing"

	"gorgonia.org/tensor"
)

// newRandomFloat32Matrix creates a 2D float32 dense tensor with the given
// shape and deterministic pseudo-random contents.
func newRandomFloat32Matrix(t *testing.T, rows, cols int, r *rand.Rand) *tensor.Dense {
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

// newZeroFloat32Matrix creates a zero-initialized 2D float32 dense tensor.
func newZeroFloat32Matrix(rows, cols int) *tensor.Dense {
	return tensor.New(
		tensor.WithShape(rows, cols),
		tensor.WithBacking(make([]float32, rows*cols)),
	)
}

// equalApprox reports whether two float32 slices are equal within a tolerance.
func equalApprox(a, b []float32, tol float32) bool {
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

// extractFloat32Backing is a helper to get the []float32 backing of a dense tensor.
func extractFloat32Backing(t *testing.T, d *tensor.Dense) []float32 {
	t.Helper()
	data, ok := d.Data().([]float32)
	if !ok {
		t.Fatalf("expected []float32 backing, got %T", d.Data())
	}
	return data
}

// Test that for supported inputs, MPSEng.MatMul produces the same results
// as StdEng.MatMul (within a small numerical tolerance).
func TestMPSEngMatMulSupportedMatchesStdEng(t *testing.T) {
	r := rand.New(rand.NewSource(1))

	const (
		m = 4
		k = 3
		n = 5
	)

	a := newRandomFloat32Matrix(t, m, k, r)
	b := newRandomFloat32Matrix(t, k, n, r)

	cCPU := newZeroFloat32Matrix(m, n)
	cMPS := newZeroFloat32Matrix(m, n)

	var cpu tensor.StdEng
	if err := cpu.MatMul(a, b, cCPU); err != nil {
		t.Fatalf("StdEng.MatMul error: %v", err)
	}

	mpsEng := NewMPSEng()
	if err := mpsEng.MatMul(a, b, cMPS); err != nil {
		t.Fatalf("MPSEng.MatMul error: %v", err)
	}

	got := extractFloat32Backing(t, cMPS)
	want := extractFloat32Backing(t, cCPU)

	if !equalApprox(got, want, 1e-4) {
		t.Fatalf("MPSEng.MatMul result differs from StdEng.MatMul.\n got:  %v\n want: %v", got, want)
	}
}

// Test that for unsupported dtypes (e.g. float64), MPSEng.MatMul falls back
// to CPU and returns the same result as StdEng.MatMul.
func TestMPSEngMatMulUnsupportedDtypeFallback(t *testing.T) {
	const (
		m = 3
		k = 2
		n = 4
	)

	// Use deterministic contents for reproducibility.
	aData := make([]float64, m*k)
	bData := make([]float64, k*n)
	for i := range aData {
		aData[i] = float64(i) + 0.5
	}
	for i := range bData {
		bData[i] = float64(i) - 0.25
	}

	a := tensor.New(
		tensor.WithShape(m, k),
		tensor.WithBacking(aData),
	)
	b := tensor.New(
		tensor.WithShape(k, n),
		tensor.WithBacking(bData),
	)

	cCPU := tensor.New(
		tensor.WithShape(m, n),
		tensor.WithBacking(make([]float64, m*n)),
	)
	cMPS := tensor.New(
		tensor.WithShape(m, n),
		tensor.WithBacking(make([]float64, m*n)),
	)

	var cpu tensor.StdEng
	if err := cpu.MatMul(a, b, cCPU); err != nil {
		t.Fatalf("StdEng.MatMul (float64) error: %v", err)
	}

	mpsEng := NewMPSEng()
	if err := mpsEng.MatMul(a, b, cMPS); err != nil {
		t.Fatalf("MPSEng.MatMul (float64) error: %v", err)
	}

	got, ok := cMPS.Data().([]float64)
	if !ok {
		t.Fatalf("expected []float64 backing for cMPS, got %T", cMPS.Data())
	}
	want, ok := cCPU.Data().([]float64)
	if !ok {
		t.Fatalf("expected []float64 backing for cCPU, got %T", cCPU.Data())
	}

	if len(got) != len(want) {
		t.Fatalf("result length mismatch: got %d, want %d", len(got), len(want))
	}
	for i := range got {
		if got[i] != want[i] {
			t.Fatalf("result mismatch at index %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

// Test that a clear shape mismatch between A and B results in a descriptive error.
func TestMPSEngMatMulShapeMismatchAB(t *testing.T) {
	mpsEng := NewMPSEng()

	// A: (2 x 3), B: (4 x 5) -> inner dims 3 vs 4 mismatch
	a := newZeroFloat32Matrix(2, 3)
	b := newZeroFloat32Matrix(4, 5)
	c := newZeroFloat32Matrix(2, 5)

	err := mpsEng.MatMul(a, b, c)
	if err == nil {
		t.Fatalf("expected error for shape-mismatched MatMul, got nil")
	}

	if got := err.Error(); got == "" {
		t.Fatalf("expected non-empty error message for shape mismatch")
	}
}

// Test that a mismatch between the implied result shape and prealloc's shape
// is reported as an error.
func TestMPSEngMatMulPreallocShapeMismatch(t *testing.T) {
	mpsEng := NewMPSEng()

	// A: (2 x 3), B: (3 x 4) -> result should be (2 x 4), but we give (2 x 3).
	a := newZeroFloat32Matrix(2, 3)
	b := newZeroFloat32Matrix(3, 4)
	c := newZeroFloat32Matrix(2, 3) // incorrect shape

	err := mpsEng.MatMul(a, b, c)
	if err == nil {
		t.Fatalf("expected error for prealloc shape mismatch, got nil")
	}

	if got := err.Error(); got == "" {
		t.Fatalf("expected non-empty error message for prealloc shape mismatch")
	}
}

// --- Benchmarks ------------------------------------------------------------

// benchmarkMatMul is a helper that benchmarks either StdEng or MPSEng MatMul
// on a single matrix size configuration.
func benchmarkMatMul(b *testing.B, m, k, n int, useMPS bool) {
	b.Helper()

	r := rand.New(rand.NewSource(42))

	a := newRandomFloat32Matrix(&testing.T{}, m, k, r)
	bMat := newRandomFloat32Matrix(&testing.T{}, k, n, r)
	c := newZeroFloat32Matrix(m, n)

	// Choose engine.
	var (
		cpu    tensor.StdEng
		mpsEng = NewMPSEng()
	)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if useMPS {
			if err := mpsEng.MatMul(a, bMat, c); err != nil {
				b.Fatalf("MPSEng.MatMul error: %v", err)
			}
		} else {
			if err := cpu.MatMul(a, bMat, c); err != nil {
				b.Fatalf("StdEng.MatMul error: %v", err)
			}
		}
	}
}

func BenchmarkStdEngMatMul_128x128(b *testing.B) {
	benchmarkMatMul(b, 128, 128, 128, false)
}

func BenchmarkMPSEngMatMul_128x128(b *testing.B) {
	benchmarkMatMul(b, 128, 128, 128, true)
}

func BenchmarkStdEngMatMul_512x512(b *testing.B) {
	benchmarkMatMul(b, 512, 512, 512, false)
}

func BenchmarkMPSEngMatMul_512x512(b *testing.B) {
	benchmarkMatMul(b, 512, 512, 512, true)
}

func BenchmarkStdEngMatMul_1024x1024(b *testing.B) {
	benchmarkMatMul(b, 1024, 1024, 1024, false)
}

func BenchmarkMPSEngMatMul_1024x1024(b *testing.B) {
	benchmarkMatMul(b, 1024, 1024, 1024, true)
}
