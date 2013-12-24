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
