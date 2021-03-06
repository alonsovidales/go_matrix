package mt

import (
	"testing"
)

func TestMatrixMultiplication(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3},
		[]float64{1, 4},
		[]float64{2, 9},
	}

	expectedRes := [][]float64{
		[]float64{10, 26},
		[]float64{37, 110},
	}

	result := Mult(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixSum(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{5, 5, 5},
		[]float64{10, 9, 14},
	}

	result := Sum(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMultTrans(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := Mult(m1, Trans(m2))
	result := MultTrans(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMultElems(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{6, 6, 4},
		[]float64{9, 20, 49},
	}

	result := MultElems(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixSub(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}
	m2 := [][]float64{
		[]float64{2, 3, 4},
		[]float64{1, 4, 7},
	}

	expectedRes := [][]float64{
		[]float64{1, -1, -3},
		[]float64{8, 1, 0},
	}

	result := Sub(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestMatrixTrans(t *testing.T) {
	m1 := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}

	expectedRes := [][]float64{
		[]float64{3, 9},
		[]float64{2, 5},
		[]float64{1, 7},
	}

	result := Trans(m1)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestSumAll(t *testing.T) {
	m := [][]float64{
		[]float64{3, 2, 1},
		[]float64{9, 5, 7},
	}

	result := SumAll(m)

	if result != 27 {
		t.Error("Expected result for SumAll: 27 but obtained:", result)
	}
}

func TestApply(t *testing.T) {
	m := [][]float64{
		[]float64{4, 2, 1},
		[]float64{8, 3, 6},
	}

	expectedRes := [][]float64{
		[]float64{2, 1, 0.5},
		[]float64{4, 1.5, 3},
	}

	f := func(x float64) float64 {
		return x / 2
	}

	result := Apply(m, f)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestDet(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	d := Det(m)

	if d != -1 {
		t.Error("Expected Det result -1,", d, "obtained")
	}

	m = [][]float64{
		[]float64{3, 3},
		[]float64{5, 2},
	}

	d = Det(m)

	if d != -9 {
		t.Error("Expected Det result -4,", d, "obtained")
	}

	m = [][]float64{
		[]float64{3, 2, 1, 9},
		[]float64{2, 7, 5, 4},
		[]float64{3, 2, 9, 7},
		[]float64{4, 3, 3, 1},
	}

	d = Det(m)

	if d != -176 {
		t.Error("Expected Det result -176,", d, "obtained")
	}
}

func TestMinors(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{-2, 0, 1},
		[]float64{9, 2, -3},
		[]float64{5, 1, -2},
	}

	result := Minors(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestCofactor(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, -1, -2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{1, -3, 1},
		[]float64{-1, -1, 2},
		[]float64{2, -3, 4},
	}

	result := Cofactors(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestInv(t *testing.T) {
	m := [][]float64{
		[]float64{1, 3, 1},
		[]float64{1, 1, 2},
		[]float64{2, 3, 4},
	}

	expectedRes := [][]float64{
		[]float64{2, 9, -5},
		[]float64{0, -2, 1},
		[]float64{-1, -3, 2},
	}

	result := Inv(m)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result[0]); j++ {
			if result[i][j] != expectedRes[i][j] {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestDiv(t *testing.T) {
	m1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	m2 := [][]float64{
		[]float64{4, 5, 6},
		[]float64{6, 5, 4},
		[]float64{4, 6, 5},
	}

	expectedRes := [][]float64{
		[]float64{7.0/10.0, 3.0/10.0, 0},
		[]float64{-3.0/10.0, 7.0/10.0, 0},
		[]float64{6.0/5.0, 1.0/5.0, -1},
	}

	result := Div(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] - expectedRes[i][j] > 0.000001 {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

func TestConcat(t *testing.T) {
	m1 := [][]float64{
		[]float64{1, 2, 3},
		[]float64{3, 2, 1},
		[]float64{2, 1, 3},
	}

	m2 := [][]float64{
		[]float64{5, 6},
		[]float64{5, 4},
		[]float64{6, 5},
	}

	expectedRes := [][]float64{
		[]float64{1, 2, 3, 5, 6},
		[]float64{3, 2, 1, 5, 4},
		[]float64{2, 1, 3, 6, 5},
	}

	result := Concat(m1, m2)

	for i := 0; i < len(result); i++ {
		for j := 0; j < len(result); j++ {
			if result[i][j] - expectedRes[i][j] > 0.000001 {
				t.Error("Expected result on pos:", i, j, ":", expectedRes[i][j], "but obtained:", result[i][j])
			}
		}
	}
}

