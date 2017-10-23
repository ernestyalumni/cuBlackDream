/**
 * @file   : activationf.h
 * @brief  : activation functions header file in CUDA C++14, 
 * @details : activation functions; 
 * 6. Using Separate Compilation in CUDA
 * The code changes required for separate compilation of device code are the same as what you already do for host code, namely using extern and static to control the visibility of symbols. Note that previously extern was ignored in CUDA code; now it will be honored
 * Read more at: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ixzz4wK3JBFup
 * 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171019
 * @ref    : wikipedia article for activation function
 *				http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#examples 
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
 * nvcc -std=c++14 -dc activationf.cu -o activationf.o
 * 
 * */
#ifndef __ACTIVATIONF_H__
#define __ACTIVATIONF_H__ 

#include <type_traits>	// std::add_pointer 

// 0
extern __device__ float identity(float);

extern __device__ float D_identity(float);

// 1
extern __device__ float sigmoid(float);

__global__ void sigmoid_kernel(const int, float*);

extern __device__ float D_sigmoid(float);

__global__ void D_sigmoid_kernel(const int, const float*, float*);

// 2
extern __device__ float tanh_overloaded(float);

__global__ void tanh_kernel(const int, float*);

extern __device__ float D_tanh(float);

__global__ void D_tanh_kernel(const int, const float*, float*);

// 3
extern __device__ float arctan_overloaded(float);

__global__ void arctan_kernel(const int, float*);

extern __device__ float D_arctan(float);

__global__ void D_arctan_kernel(const int, const float*, float*);


// 4
extern __device__ float ReLU(float);

__global__ void ReLU_kernel(const int, float*);

extern __device__ float D_ReLU(float);

__global__ void D_ReLU_kernel(const int, const float*, float*);

// 5
extern __device__ float Gaussian(float);


/**
 * 	@fn Gaussian_kernel
 * 	@param c 
 *  @note exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */
__global__ void Gaussian_kernel(const int, const int, const int, float*);

extern __device__ float D_Gaussian(float);


/* move to Axon.h*/
// function pointer type for __device__ activation functions
// pf = processing function
//using activat_pf = std::add_pointer<float(float)>::type;

// array of function pointers pointing to activation functions
/** @fn d_activat_fs 
 * 	@brief d_activate_fs, d, on device GPU, activation functions, as an array of them 
 * */
//extern __device__ activat_pf d_activat_fs[6] = { identity, sigmoid, tanh_overloaded, arctan_overloaded, ReLU, Gaussian };

/** @fn D_activat_fs 
 * 	@brief D_activate_fs, D, derivatives or gradient, activation functions, as an array of them 
 * */
//extern __device__ activat_pf D_activat_fs[6] = { D_identity, D_sigmoid, D_tanh, D_arctan, D_ReLU, D_Gaussian };




/** 
 * 	@fn D_Gaussian_kernel
 * 	@brief derivative of Gaussian_kernel
 * 	@param c 
 * 	@param sigma_dev
 *  @note -(z-c) / ( sigma_dev*sigma_dev) * exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */	
__global__ void D_Gaussian_kernel(const float, const float, const int, const float*, float*);





#endif // END of __ACTIVATIONF_H__
