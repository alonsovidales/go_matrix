package mt

const KER_MATRIX_ADD_BIAS_TOP = `
//
// Generated by NVIDIA NVVM Compiler
// Compiler built on Thu Mar 13 19:31:35 2014 (1394735495)
// Cuda compilation tools, release 6.0, V6.0.1
//

.version 4.0
.target sm_20
.address_size 64


.visible .entry matrixAddBiasTop(
	.param .u64 matrixAddBiasTop_param_0,
	.param .u64 matrixAddBiasTop_param_1,
	.param .u32 matrixAddBiasTop_param_2,
	.param .u32 matrixAddBiasTop_param_3,
	.param .u32 matrixAddBiasTop_param_4,
	.param .u32 matrixAddBiasTop_param_5
)
{
	.reg .pred 	%p<5>;
	.reg .s32 	%r<13>;
	.reg .s64 	%rd<10>;
	.reg .f64 	%fd<2>;


	ld.param.u64 	%rd2, [matrixAddBiasTop_param_0];
	ld.param.u64 	%rd3, [matrixAddBiasTop_param_1];
	ld.param.u32 	%r3, [matrixAddBiasTop_param_2];
	ld.param.u32 	%r4, [matrixAddBiasTop_param_3];
	ld.param.u32 	%r5, [matrixAddBiasTop_param_4];
	ld.param.u32 	%r6, [matrixAddBiasTop_param_5];
	mov.u32 	%r7, %ctaid.x;
	mov.u32 	%r8, %tid.x;
	mad.lo.s32 	%r9, %r7, %r4, %r8;
	mov.u32 	%r10, %ctaid.y;
	mov.u32 	%r11, %tid.y;
	mad.lo.s32 	%r1, %r10, %r5, %r11;
	mad.lo.s32 	%r2, %r1, %r3, %r9;
	setp.lt.s32	%p1, %r2, %r6;
	setp.lt.s32	%p2, %r9, %r3;
	and.pred  	%p3, %p1, %p2;
	@!%p3 bra 	BB0_4;
	bra.uni 	BB0_1;

BB0_1:
	cvta.to.global.u64 	%rd4, %rd2;
	mul.wide.s32 	%rd5, %r2, 8;
	add.s64 	%rd1, %rd4, %rd5;
	setp.eq.s32	%p4, %r1, 0;
	@%p4 bra 	BB0_3;

	cvta.to.global.u64 	%rd6, %rd3;
	sub.s32 	%r12, %r2, %r3;
	mul.wide.s32 	%rd7, %r12, 8;
	add.s64 	%rd8, %rd6, %rd7;
	ld.global.f64 	%fd1, [%rd8];
	st.global.f64 	[%rd1], %fd1;
	bra.uni 	BB0_4;

BB0_3:
	mov.u64 	%rd9, 4607182418800017408;
	st.global.u64 	[%rd1], %rd9;

BB0_4:
	ret;
}`
