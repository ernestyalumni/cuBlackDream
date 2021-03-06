/**
 * @file   : activationf.cu
 * @brief  : activation functions content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171020  
 * @ref    :  
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * nvcc -std=c++14 -lcublas -dc Axon.cu -o Axon.o
 * 
 * */
#include "activationf.h"

// 0 
__global__ void identity_kernel(const int SIZE, float*z) {
}

__global__ void D_identity_kernel(const int SIZE, const float* z, float* d_a) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{	
		d_a[tid]= 1.0f;	
	}
}

// 1
__global__ void sigmoid_kernel(const int SIZE, float*z) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{	
		float a_val = z[tid];

		a_val = 1.f/(1.f + expf(-a_val));

		z[tid]=a_val;
	}
}


__global__ void D_sigmoid_kernel(const int SIZE, const float* z, float* d_a) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{	
		float a_val = z[tid];

		a_val = 1.f/(1.f + expf(-a_val));
		a_val = a_val * ( 1.0f - a_val );
	
		d_a[tid]=a_val;	
	}
}


// 2
__global__ void tanh_kernel(const int SIZE, float*z) 
{
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{	
		float a_val = z[tid];

		a_val = tanhf(a_val);
		z[tid] = a_val;
	}
}


__global__ void D_tanh_kernel(const int SIZE, const float* z, float*d_a) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];

		a_val = tanhf(a_val);
		a_val = 1.0f - (a_val)*a_val;	
		d_a[tid] = a_val;	
	}
}



__global__ void arctan_kernel(const int SIZE, float*z) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];
	
		a_val = atanf(a_val);
		z[tid] = a_val;
	}
}



__global__ void D_arctan_kernel(const int SIZE, const float* z, float*d_a) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];

		a_val = 1.0f / ( 1.0f + a_val*a_val);
		d_a[tid] = a_val;
	}
}


__global__ void ReLU_kernel(const int SIZE, float*z) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];
	
		if (a_val < 0.f) { 
			z[tid] = 0.f;
		} 
	}
}


__global__ void D_ReLU_kernel(const int SIZE, const float*z, float*d_a) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];
	
		if (a_val < 0.f) { 
			d_a[tid] = 0.f;
		} else {
			d_a[tid] = 1.0f; 
		}
	}	
}
	

/**
 * 	@fn Gaussian_kernel
 * 	@param c 
 * 	@param sigma_dev
 *  @note exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */
__global__ void Gaussian_kernel(const int SIZE, float* z) {
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];
	
		a_val = expf( -1.0f * ( a_val - 0.f)*(a_val-0.f) / 2.0f / (1.f*1.f) ) ;
	
		z[tid] = a_val;
	}
}



/** 
 * 	@fn D_Gaussian_kernel
 * 	@brief derivative of Gaussian_kernel
 * 	@param c 
 * 	@param sigma_dev
 *  @note -(z-c) / ( sigma_dev*sigma_dev) * exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */	
__global__ void D_Gaussian_kernel(const int SIZE, const float* z, float*d_a) 
{
	int kx = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (kx >= SIZE) { return; } 

	for (int tid=kx; tid < SIZE; tid += gridDim.x*blockDim.x) 
	{
		float a_val = z[tid];
	
		a_val = -1.0f * (a_val - 0.f)/(1.f*1.f) * 
			expf( -1.0f * ( a_val - 0.f)*(a_val-0.f) / 2.0f / (1.f*1.f) ) ;

		d_a[tid] = a_val;
	}
}


