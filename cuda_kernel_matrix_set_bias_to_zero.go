package mt

const KER_MATRIX_SET_BIAS_TO_ZERO = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Thu Mar 13 19:31:35 2014 (1394735495)
// Cuda compilation tools, release 6.0, V6.0.1
//

.version 4.0
.target sm_20
.address_size 64


.visible .entry matrixSetBiasToZero(
	.param .u64 matrixSetBiasToZero_param_0,
	.param .u32 matrixSetBiasToZero_param_1,
	.param .u32 matrixSetBiasToZero_param_2,
	.param .u32 matrixSetBiasToZero_param_3
)
{
	.reg .pred 	%p<2>;
	.reg .s32 	%r<8>;
	.reg .s64 	%rd<6>;


	ld.param.u64 	%rd1, [matrixSetBiasToZero_param_0];
	ld.param.u32 	%r3, [matrixSetBiasToZero_param_1];
	ld.param.u32 	%r2, [matrixSetBiasToZero_param_2];
	ld.param.u32 	%r4, [matrixSetBiasToZero_param_3];
	mov.u32 	%r5, %ctaid.y;
	mov.u32 	%r6, %tid.y;
	mad.lo.s32 	%r1, %r5, %r4, %r6;
	setp.ge.s32	%p1, %r1, %r3;
	@%p1 bra 	BB0_2;

	cvta.to.global.u64 	%rd2, %rd1;
	mul.lo.s32 	%r7, %r1, %r2;
	mul.wide.s32 	%rd3, %r7, 8;
	add.s64 	%rd4, %rd2, %rd3;
	mov.u64 	%rd5, 0;
	st.global.u64 	[%rd4], %rd5;

BB0_2:
	ret;
}`