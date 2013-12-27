// Package mt
// Linear algebra functions to work with matrix

package mt

// Returns the subtraction of the given two matrix
func Sub(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] - m2[x][y]
		}
	}

	return
}

// Returns the sum of the given two matrix
func Sum(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] + m2[x][y]
		}
	}

	return
}

// Calculates the determinant of the matrix
func Det(m [][]float64) (rm float64) {
	// Sum diagonals
	ml := len(m)
	sums := make([]float64, ml*2)
	for i := 0; i < len(sums); i++ {
		sums[i] = 1
	}

	for r := 0; r < ml; r++ {
		for c := 0; c < ml; c++ {
			if c-r < 0 {
				sums[ml+c-r] *= m[c][r]
			} else {
				sums[c-r] *= m[c][r]
			}

			if c+r >= ml {
				sums[c+r] *= m[c][r]
			} else {
				sums[c+r+ml] *= m[c][r]
			}
		}
	}

	to := len(sums)
	if ml == 2 {
		to = 2
		ml = 1
	}
	for i := 0; i < to; i++ {
		if i >= ml {
			rm -= sums[i]
		} else {
			rm += sums[i]
		}
	}
	return
}

// Returns the minors matrix
func Minors(m [][]float64) (rm [][]float64) {
	ml := len(m)
	rm = make([][]float64, ml)
	for r := 0; r < ml; r++ {
		rm[r] = make([]float64, ml)
		for c := 0; c < ml; c++ {
			auxM := [][]float64{}
			for ra := 0; ra < ml; ra++ {
				if ra != r {
					auxR := []float64{}
					for ca := 0; ca < ml; ca++ {
						if ca != c {
							auxR = append(auxR, m[ra][ca])
						}
					}
					auxM = append(auxM, auxR)
				}
			}
			rm[r][c] = Det(auxM)
		}
	}

	return
}

// Returns the cofactors matrix
func Cofactors(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))
	for r := 0; r < len(m); r++ {
		rm[r] = make([]float64, len(m[0]))
		for c := 0; c < len(m[0]); c++ {
			if (c+r)%2 == 0 {
				rm[r][c] = m[r][c]
			} else {
				rm[r][c] = -m[r][c]
			}
		}
	}

	return
}

// Calculates the inverse matrix
func Inv(m [][]float64) (rm [][]float64) {
	dm := Det(m)
	adj := Trans(Cofactors(Minors(m)))

	rm = MultBy(adj, float64(1)/dm)

	return
}

// Divide the first matrix by the second one
func Div(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	return Mult(m1, Inv(m2))
}

// Returns the rm of multiply all the elements of a matrix by a float number
func MultBy(m1 [][]float64, n float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * n
		}
	}

	return
}

// Returns the rm of multiply the given two matrix
func Mult(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m2[0]))

		for y := 0; y < len(m2[0]); y++ {
			for k := 0; k < len(m2); k++ {
				rm[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

// Returns as rm a matrix with all the elemtns of the first one multiplyed
// by the second one elems
func MultElems(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float64, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * m2[x][y]
		}
	}

	return
}

// Matrix Transpose, returns the transpose of the given square matrix
func Trans(m1 [][]float64) (rm [][]float64) {
	if len(m1) == 0 {
		return [][]float64{}
	}
	rm = make([][]float64, len(m1[0]))

	// Initialize the matrix
	for x := 0; x < len(m1[0]); x++ {
		rm[x] = make([]float64, len(m1))

		for y := 0; y < len(m1); y++ {
			rm[x][y] = m1[y][x]
		}
	}

	return
}

// Sum all the elements in a matrix
func SumAll(m [][]float64) (rm float64) {
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			rm += m[i][j]
		}
	}

	return
}

// Apply a function to all the elements of a matrix, the function will receive a
// float64 as param and returns a float64 too
func Apply(m [][]float64, f func(x float64) float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	// Initialize the matrix
	for x := 0; x < len(m); x++ {
		rm[x] = make([]float64, len(m[0]))
		for y := 0; y < len(m[0]); y++ {
			rm[x][y] = f(m[x][y])
		}
	}

	return
}

// Returns a copy of the matrix
func Copy(m [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m))

	for i := 0; i < len(m); i++ {
		rm[i] = make([]float64, len(m[i]))
		for j := 0; j < len(m[i]); j++ {
			rm[i][j] = m[i][j]
		}
	}

	return
}

// Concatenates two matrix elements, ex:
// m1 = (M111, M112, M113)
//      (M121, M122, M123)
//      (M131, M132, M133)
// m2 = (M211, M212, M213)
//      (M221, M222, M223)
//      (M231, M232, M233)
// rm = (M111, M112, M113, M221, M222, M223)
//      (M121, M122, M123, M221, M222, M223)
//      (M131, M132, M133, M231, M232, M233)
func Concat(m1 [][]float64, m2 [][]float64) (rm [][]float64) {
	rm = make([][]float64, len(m1))
	for i := 0; i < len(m1); i++ {
		rm[i] = make([]float64, len(m1[i]) + len(m2[i]))
		for j := 0; j < len(m1[i]); j++ {
			rm[i][j] = m1[i][j]
		}
		for j := 0; j < len(m2[i]); j++ {
			rm[i][j + len(m1[i])] = m2[i][j]
		}
	}

	return
}
