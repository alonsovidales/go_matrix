// Package mt, implements some simple functions to work with matrix

package mt

// Returns the subtraction of the given two matrix
func Sub(m1 [][]float64, m2 [][]float64) (result [][]float64) {
	result = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		result[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			result[x][y] = m1[x][y] - m2[x][y]
		}
	}

	return
}

// Returns the sum of the given two matrix
func Sum(m1 [][]float64, m2 [][]float64) (result [][]float64) {
	result = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		result[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			result[x][y] = m1[x][y] + m2[x][y]
		}
	}

	return
}

// Returns the result of multiply the given two matrix
func Mult(m1 [][]float64, m2 [][]float64) (result [][]float64) {
	result = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		result[x] = make([]float64, len(m1))

		for y := 0; y < len(m1); y++ {
			for k := 0; k < len(m2); k++ {
				result[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

// Matrix Transpose, returns the transpose of the given 2x2 matrix
func Trans(m1 [][]float64) (result [][]float64) {
	result = make([][]float64, len(m1[0]))

	// Initialize the matrix
	for x := 0; x < len(m1[0]); x++ {
		result[x] = make([]float64, len(m1))

		for y := 0; y < len(m1); y++ {
			result[x][y] = m1[y][x]
		}
	}

	return
}
