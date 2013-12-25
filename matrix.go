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
		result[x] = make([]float64, len(m2[0]))

		for y := 0; y < len(m2[0]); y++ {
			for k := 0; k < len(m2); k++ {
				result[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

// Returns as result a matrix with all the elemtns of the first one multiplyed
// by the second one elems
func MultElems(m1 [][]float64, m2 [][]float64) (result [][]float64) {
	result = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		result[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			result[x][y] = m1[x][y] * m2[x][y]
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

// Sum all the elements in a matrix
func SumAll(m [][]float64) (result float64) {
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			result += m[i][j]
		}
	}

	return
}

// Apply a function to all the elements of a matrix, the function will receive a
// float64 as param and return a float64 too
func Apply(m [][]float64, f func(x float64) (float64)) (result [][]float64) {
	result = make([][]float64, len(m))

	// Initialize the matrix
	for x := 0; x < len(m); x++ {
		result[x] = make([]float64, len(m[0]))
		for y := 0; y < len(m[0]); y++ {
			result[x][y] = f(m[x][y])
		}
	}

	return
}
