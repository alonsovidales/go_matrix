package mt

const KER_MATRIX_POW_TWO = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Thu Mar 13 19:31:35 2014 (1394735495)
// Cuda compilation tools, release 6.0, V6.0.1
//

.version 4.0
.target sm_20
.address_size 64


.visible .entry matrixPowTwo(
	.param .u64 matrixPowTwo_param_0,
	.param .u32 matrixPowTwo_param_1,
	.param .u32 matrixPowTwo_param_2,
	.param .u32 matrixPowTwo_param_3,
	.param .u32 matrixPowTwo_param_4
)
{
	.reg .pred 	%p<4>;
	.reg .s32 	%r<12>;
	.reg .s64 	%rd<5>;
	.reg .f64 	%fd<3>;


	ld.param.u64 	%rd1, [matrixPowTwo_param_0];
	ld.param.u32 	%r2, [matrixPowTwo_param_1];
	ld.param.u32 	%r3, [matrixPowTwo_param_2];
	ld.param.u32 	%r4, [matrixPowTwo_param_3];
	ld.param.u32 	%r5, [matrixPowTwo_param_4];
	mov.u32 	%r6, %ctaid.x;
	mov.u32 	%r7, %tid.x;
	mad.lo.s32 	%r8, %r6, %r2, %r7;
	mov.u32 	%r9, %ctaid.y;
	mov.u32 	%r10, %tid.y;
	mad.lo.s32 	%r11, %r9, %r3, %r10;
	mad.lo.s32 	%r1, %r11, %r4, %r8;
	setp.lt.s32	%p1, %r1, %r5;
	setp.lt.s32	%p2, %r8, %r4;
	and.pred  	%p3, %p1, %p2;
	@!%p3 bra 	BB0_2;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd2, %rd1;
	mul.wide.s32 	%rd3, %r1, 8;
	add.s64 	%rd4, %rd2, %rd3;
	ld.global.f64 	%fd1, [%rd4];
	mul.f64 	%fd2, %fd1, %fd1;
	st.global.f64 	[%rd4], %fd2;

BB0_2:
	ret;
}`
