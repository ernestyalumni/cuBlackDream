/**
 * @file   : activationf.h
 * @brief  : activation functions header file in CUDA C++14, 
 * @details : activation functions 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171019
 * @ref    : wikipedia article for activation function
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
 * nvcc -std=c++14 -lcublas -dc activationf.cu -o activationf.o
 * 
 * */

#ifndef __ACTIVATIONF_H__
#define __ACTIVATIONF_H__ 

__global__ void sigmoid_kernel(const int, float*);

__global__ void D_sigmoid_kernel(const int, const float*, float*);

__global__ void tanh_kernel(const int, float*);

__global__ void D_tanh_kernel(const int, const float*, float*);

__global__ void arctan_kernel(const int, float*);

__global__ void D_arctan_kernel(const int, const float*, float*);

__global__ void ReLU_kernel(const int, float*);

__global__ void D_ReLU_kernel(const int, const float*, float*);

/**
 * 	@fn Gaussian_kernel
 * 	@param c 
 *  @note exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */
__global__ void Gaussian_kernel(const int, const int, const int, float*);



/** 
 * 	@fn D_Gaussian_kernel
 * 	@brief derivative of Gaussian_kernel
 * 	@param c 
 * 	@param sigma_dev
 *  @note -(z-c) / ( sigma_dev*sigma_dev) * exp(-(z-c)^2 / (2.f * sigma_dev*sigma_dev) )
 * */	
__global__ void D_Gaussian_kernel(const int, const int, const int, const float*, float*);



#endif // END of __ACTIVATIONF_H__
