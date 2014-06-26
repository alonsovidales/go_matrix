// Package mt
// Linear algebra functions to work with matrix

package mt

import (
	"math"
	"fmt"
	"github.com/barnex/cuda5/cu"
	"unsafe"
)

type CudaMatrix struct {
	m cu.DevicePtr
	w int
	h int
}

var currentBuff = ""
var userMem = make(map[string]map[cu.DevicePtr]bool)

const DEBUG = false
const CUDA_DEVICE = 0

var multMod, subMod, addMod, multAllMod, negMatrixMod, multTransMod, multByMod,
	sigmoidMatrixMod, logMatrixMod, oneMinusMod, addBiasMod, removeBias, powTwoMod, sigmoidGradMod cu.Function
var maxNumThreads int
var cudaInitialized = false
var ctx cu.Context
var dev cu.Device

func InitCuda() {
	if !cudaInitialized {
		var mod cu.Module

		cu.Init(0)
		dev = cu.DeviceGet(CUDA_DEVICE)
		maxNumThreads = dev.Attribute(cu.MAX_THREADS_PER_BLOCK)

		ctx = cu.CtxCreate(cu.CTX_SCHED_AUTO, CUDA_DEVICE)
		ctx.SetCurrent()

		if DEBUG {
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult.ptx")
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sub.ptx")
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add.ptx")
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_all.ptx")
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoad("/cuda_modules/matrix_neg.ptx")
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_trans.ptx")
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid.ptx")
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoad("/cuda_modules/matrix_log.ptx")
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoad("/cuda_modules/matrix_one_minus.ptx")
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoad("/cuda_modules/matrix_add_bias.ptx")
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_remove_bias.ptx")
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoad("/cuda_modules/matrix_pow_two.ptx")
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoad("/cuda_modules/matrix_sigmoid_gradient.ptx")
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoad("/cuda_modules/matrix_mult_by.ptx")
			multByMod = mod.GetFunction("matrixMultBy")
		} else {
			mod = cu.ModuleLoadData(KER_MATRIX_MULT)
			multMod = mod.GetFunction("matrixMul")
			mod = cu.ModuleLoadData(KER_MATRIX_SUB)
			subMod = mod.GetFunction("matrixSub")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD)
			addMod = mod.GetFunction("matrixAdd")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_ALL)
			multAllMod = mod.GetFunction("matrixMultAll")
			mod = cu.ModuleLoadData(KER_MATRIX_NEG)
			negMatrixMod = mod.GetFunction("matrixNeg")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_TRANS)
			multTransMod = mod.GetFunction("matrixMulTrans")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID)
			sigmoidMatrixMod = mod.GetFunction("matrixSigmoid")
			mod = cu.ModuleLoadData(KER_MATRIX_LOG)
			logMatrixMod = mod.GetFunction("matrixLog")
			mod = cu.ModuleLoadData(KER_MATRIX_ONE_MINUS)
			oneMinusMod = mod.GetFunction("matrixOneMinus")
			mod = cu.ModuleLoadData(KER_MATRIX_ADD_BIAS)
			addBiasMod = mod.GetFunction("matrixAddBias")
			mod = cu.ModuleLoadData(KER_MATRIX_REMOVE_BIAS)
			removeBias = mod.GetFunction("matrixRemoveBias")
			mod = cu.ModuleLoadData(KER_MATRIX_POW_TWO)
			powTwoMod = mod.GetFunction("matrixPowTwo")
			mod = cu.ModuleLoadData(KER_MATRIX_SIGMOID_GRADIENT)
			sigmoidGradMod = mod.GetFunction("matrixSigmoidGrad")
			mod = cu.ModuleLoadData(KER_MATRIX_MULT_BY)
			multByMod = mod.GetFunction("matrixMultBy")
		}

		cudaInitialized = true
	}

	// Ugly hack to prevent problems with the libraries and the context
	// handling
	if cu.CtxGetCurrent() == 0 && ctx != 0 {
		ctx.SetCurrent()
	}
}

func StartBufferingMem(buff string) {
	currentBuff = buff
	userMem[buff] = make(map[cu.DevicePtr]bool)
}

func AddToBuff(ptr cu.DevicePtr) {
	if currentBuff != "" {
		userMem[currentBuff][ptr] = true
	}
}

func FreeMem(buff string) {
	for m, _ := range(userMem[currentBuff]) {
		cu.MemFree(m)
		delete(userMem[currentBuff], m)
	}
}

func (p *CudaMatrix) Free() {
	delete(userMem[currentBuff], p.m)
	cu.MemFree(p.m)
}

func InitCudaMatrix(w int, h int) (p *CudaMatrix) {
	size := int64(w * h) * cu.SIZEOF_FLOAT32
	InitCuda()
	if DEBUG {
		fmt.Println("InitCudaMatrix")
	}
	p = &CudaMatrix {
		w: w,
		h: h,
		m: cu.MemAlloc(size),
	}
	AddToBuff(p.m)
	// Initialize this var to zeros
	aux := make([]float32, w * h)
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&aux[0]), size)

	return
}

func (m *CudaMatrix) CudaCopyTo(t *CudaMatrix) (*CudaMatrix) {
	size := int64(m.w * m.h) * cu.SIZEOF_FLOAT32
	InitCuda()
	cu.MemcpyDtoD(t.m, m.m, size)

	return t
}

func (m *CudaMatrix) CudaCopy() (r *CudaMatrix) {
	size := int64(m.w * m.h) * cu.SIZEOF_FLOAT32

	InitCuda()
	if DEBUG {
		fmt.Println("CudaCopy")
	}
	r = &CudaMatrix {
		m: cu.MemAlloc(size),
		w: m.w,
		h: m.h,
	}
	AddToBuff(r.m)
	cu.MemcpyDtoD(r.m, m.m, size)

	return
}

func GetCudaMatrix(m [][]float32) (p *CudaMatrix) {
	if DEBUG {
		fmt.Println("GetCudaMatrix")
	}
	p = &CudaMatrix {
		w: len(m[0]),
		h: len(m),
	}

	linealM := make([]float32, len(m) * len(m[0]))
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			linealM[(i * p.w) + j] = m[i][j]
		}
	}
	size := int64(len(linealM)) * cu.SIZEOF_FLOAT32

	InitCuda()
	p.m = cu.MemAlloc(size)
	AddToBuff(p.m)
	cu.MemcpyHtoD(p.m, unsafe.Pointer(&linealM[0]), size)

	return
}

func (p *CudaMatrix) TransOneDimMatrix() (*CudaMatrix) {
	p.w ^= p.h
	p.h ^= p.w
	p.w ^= p.h

	return p
}

func (p *CudaMatrix) GetMatrixFromCuda() (m [][]float32) {
	buff := make([]float32, p.h * p.w)
	m = make([][]float32, p.h)

	InitCuda()
	cu.MemcpyDtoH(unsafe.Pointer(&buff[0]), p.m, int64(len(buff)) * cu.SIZEOF_FLOAT32)
	for i := 0; i < p.h; i++ {
		m[i] = buff[i * p.w : (i + 1) * p.w]
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaMultAllElemsTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(multAllMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaMultAllElems(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	if DEBUG {
		fmt.Println("CudaMultAllElems")
	}
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: cu.MemAlloc(int64(m2.w * m1.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(multAllMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the rm of Multiply the given two matrix
func MultElems(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMultAllElems(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func (m *CudaMatrix) SetPosTo(val float32, x int, y int) (*CudaMatrix) {
	buff := make([]float32, m.h * m.w)
	buffPoint := unsafe.Pointer(&buff[0])
	size := int64(len(buff)) * cu.SIZEOF_FLOAT32
	InitCuda()
	cu.MemcpyDtoH(buffPoint, m.m, size)
	buff[(y * m.w) + x] = val
	cu.MemcpyHtoD(m.m, buffPoint, size)

	return m
}

func (m *CudaMatrix) RemoveBiasTo(rm *CudaMatrix) (*CudaMatrix) {
	size := rm.w * rm.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(removeBias, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) RemoveBias() (rm *CudaMatrix) {
	InitCuda()

	if DEBUG {
		fmt.Println("RemoveBias")
	}
	rm = &CudaMatrix{
		w: m.w - 1,
		h: m.h,
		m: cu.MemAlloc(int64((m.w - 1) * m.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(m.m)

	size := rm.w * rm.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(removeBias, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) AddBiasTo(rm *CudaMatrix) (*CudaMatrix) {
	size := rm.w * rm.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(addBiasMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

func (m *CudaMatrix) AddBias() (rm *CudaMatrix) {
	InitCuda()

	if DEBUG {
		fmt.Println("AddBias")
	}
	rm = &CudaMatrix{
		w: m.w + 1,
		h: m.h,
		m: cu.MemAlloc(int64((m.w + 1) * m.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	size := rm.w * rm.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(addBiasMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

func (m *CudaMatrix) MultBy(by float32) (*CudaMatrix) {
	InitCuda()

	size := m.w * m.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := m.w
	resH := m.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&by),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(multByMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return m
}

func (m *CudaMatrix) applyFunc(function cu.Function) {
	InitCuda()

	size := m.w * m.h
	matrixSplits := int(math.Ceil(float64(size) / float64(maxNumThreads)))
	resW := m.w
	resH := m.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	//fmt.Println("WINDOW W:", resW, "H:", resH, "W:", m.w, "h:", m.h, "Splits:", matrixSplits)

	args := []unsafe.Pointer{
		unsafe.Pointer(&m.m),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&m.w),
		unsafe.Pointer(&size),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(function, 1, 1, 1, resW, resH, 1, 0, 0, args)
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) CudaLogMatrix() (*CudaMatrix) {
	m.applyFunc(logMatrixMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) SigmoidGradient() (*CudaMatrix) {
	m.applyFunc(sigmoidGradMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) CudaSigmoidMatrix() (*CudaMatrix) {
	m.applyFunc(sigmoidMatrixMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) CudaOneMinusMatrix() (*CudaMatrix) {
	m.applyFunc(oneMinusMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) PowTwo() (*CudaMatrix) {
	m.applyFunc(powTwoMod)

	return m
}

// Returns the rm of Multiply the given two matrix
func (m *CudaMatrix) CudaNegMatrix() (*CudaMatrix) {
	m.applyFunc(negMatrixMod)

	return m
}

func Neg(m [][]float32) (rm [][]float32) {
	cm := GetCudaMatrix(m)

	cm.CudaNegMatrix()

	rm = cm.GetMatrixFromCuda()

	return
}

// Returns as rm a matrix with all the elemtns of the first one multiplyed
// by the second one elems
func MultElemsNoCuda(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float32, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * m2[x][y]
		}
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaMult(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	if DEBUG {
		fmt.Println("CudaMult")
	}
	rm = &CudaMatrix{
		w: m2.w,
		h: m1.h,
		m: cu.MemAlloc(int64(m2.w * m1.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(multMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

func CudaMultTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&m2.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(multMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func Mult(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMult(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the rm of multiply the given two matrix without use cuda
func MultNoCuda(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float32, len(m2[0]))

		for y := 0; y < len(m2[0]); y++ {
			for k := 0; k < len(m2); k++ {
				rm[x][y] += m1[x][k] * m2[k][y]
			}
		}
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaSubTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(subMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaSub(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	if DEBUG {
		fmt.Println("CudaSub")
	}
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: cu.MemAlloc(int64(m1.w * m1.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(subMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the subtraction of the given two matrix
func Sub(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSub(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the subtraction of the given two matrix
func SubNoCuda(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float32, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] - m2[x][y]
		}
	}

	return
}

// Returns the rm of Multiply the given two matrix
func CudaSumTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	InitCuda()
	cu.LaunchKernel(addMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaSum(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	InitCuda()
	if DEBUG {
		fmt.Println("CudaSum")
	}
	rm = &CudaMatrix{
		w: m1.w,
		h: m1.h,
		m: cu.MemAlloc(int64(m1.w * m1.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(addMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

// Returns the subtraction of the given two matrix
func Sum(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaSum(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

func CudaMultTransTo(m1 *CudaMatrix, m2 *CudaMatrix, rm *CudaMatrix) (*CudaMatrix) {
	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	InitCuda()
	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(multTransMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return rm
}

// Returns the rm of Multiply the given two matrix
func CudaMultTrans(m1 *CudaMatrix, m2 *CudaMatrix) (rm *CudaMatrix) {
	if DEBUG {
		fmt.Println("CudaMultTrans")
	}
	InitCuda()
	rm = &CudaMatrix{
		w: m2.h,
		h: m1.h,
		m: cu.MemAlloc(int64(m2.h * m1.h) * cu.SIZEOF_FLOAT32),
	}
	AddToBuff(rm.m)

	matrixSplits := int(math.Ceil(float64(rm.w * rm.h) / float64(maxNumThreads)))
	resultSize := rm.w * rm.h

	resW := rm.w
	resH := rm.h
	if matrixSplits > 1 {
		resW = int(math.Ceil(float64(resW) / float64(matrixSplits)))
		resH = int(math.Ceil(float64(resH) / float64(matrixSplits)))
	}

	args := []unsafe.Pointer{
		unsafe.Pointer(&rm.m),
		unsafe.Pointer(&m1.m),
		unsafe.Pointer(&m2.m),
		unsafe.Pointer(&m1.w),
		unsafe.Pointer(&rm.w),
		unsafe.Pointer(&resW),
		unsafe.Pointer(&resH),
		unsafe.Pointer(&resultSize),
		unsafe.Pointer(&matrixSplits),
	}

	cu.LaunchKernel(multTransMod, 1, 1, 1, resW, resH, 1, 0, 0, args)

	return
}

// Multiply on matrix by the transpose of the second matrix
func MultTrans(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	cm1 := GetCudaMatrix(m1)
	cm2 := GetCudaMatrix(m2)

	result := CudaMultTrans(cm1, cm2)

	rm = result.GetMatrixFromCuda()

	return
}

// Returns the sum of the given two matrix
func SumNoCuda(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float32, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] + m2[x][y]
		}
	}

	return
}

// Calculates the determinant of the matrix
func Det(m [][]float32) (rm float32) {
	// Sum diagonals
	ml := len(m)
	sums := make([]float32, ml*2)
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
func Minors(m [][]float32) (rm [][]float32) {
	ml := len(m)
	rm = make([][]float32, ml)
	for r := 0; r < ml; r++ {
		rm[r] = make([]float32, ml)
		for c := 0; c < ml; c++ {
			auxM := [][]float32{}
			for ra := 0; ra < ml; ra++ {
				if ra != r {
					auxR := []float32{}
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
func Cofactors(m [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m))
	for r := 0; r < len(m); r++ {
		rm[r] = make([]float32, len(m[0]))
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
func Inv(m [][]float32) (rm [][]float32) {
	dm := Det(m)
	adj := Trans(Cofactors(Minors(m)))

	rm = MultBy(adj, 1.0/dm)

	return
}

// Divide the first matrix by the second one
func Div(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	return Mult(m1, Inv(m2))
}

// Returns the rm of multiply all the elements of a matrix by a float number
func MultBy(m1 [][]float32, n float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	// Initialize the matrix
	for x := 0; x < len(m1); x++ {
		rm[x] = make([]float32, len(m1[0]))

		for y := 0; y < len(m1[0]); y++ {
			rm[x][y] = m1[x][y] * n
		}
	}

	return
}

// Matrix Transpose, returns the transpose of the given square matrix
func Trans(m1 [][]float32) (rm [][]float32) {
	if len(m1) == 0 {
		return [][]float32{}
	}
	rm = make([][]float32, len(m1[0]))

	// Initialize the matrix
	for x := 0; x < len(m1[0]); x++ {
		rm[x] = make([]float32, len(m1))

		for y := 0; y < len(m1); y++ {
			rm[x][y] = m1[y][x]
		}
	}

	return
}

// Sum all the elements in a matrix
func SumAll(m [][]float32) (rm float32) {
	for i := 0; i < len(m); i++ {
		for j := 0; j < len(m[0]); j++ {
			rm += m[i][j]
		}
	}

	return
}

// Apply a function to all the elements of a matrix, the function will receive a
// float32 as param and returns a float32 too
func Apply(m [][]float32, f func(x float32) float32) (rm [][]float32) {
	rm = make([][]float32, len(m))

	// Initialize the matrix
	for x := 0; x < len(m); x++ {
		rm[x] = make([]float32, len(m[0]))
		for y := 0; y < len(m[0]); y++ {
			rm[x][y] = f(m[x][y])
		}
	}

	return
}

// Returns a copy of the matrix
func Copy(m [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m))

	for i := 0; i < len(m); i++ {
		rm[i] = make([]float32, len(m[i]))
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
func Concat(m1 [][]float32, m2 [][]float32) (rm [][]float32) {
	rm = make([][]float32, len(m1))
	for i := 0; i < len(m1); i++ {
		rm[i] = make([]float32, len(m1[i]) + len(m2[i]))
		for j := 0; j < len(m1[i]); j++ {
			rm[i][j] = m1[i][j]
		}
		for j := 0; j < len(m2[i]); j++ {
			rm[i][j + len(m1[i])] = m2[i][j]
		}
	}

	return
}
