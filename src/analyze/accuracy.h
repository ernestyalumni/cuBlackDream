/**
 * @file   : accuracy.h
 * @brief  : scoring accuracy functions header file in CUDA C++14, 
 * @details : CUDA C/C++ kernel functions (__global__) to predict; 
 * 6. Using Separate Compilation in CUDA
 * The code changes required for separate compilation of device code are the same as what you already do for host code, namely using extern and static to control the visibility of symbols. Note that previously extern was ignored in CUDA code; now it will be honored
 * Read more at: http://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#ixzz4wK3JBFup
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
 * nvcc -std=c++11 -arch='sm_61' -I./cub-1.6.4/ -dc accuracy.cu -o accuracy.o
 * 
 * */
#ifndef __ACCURACY_H__
#define __ACCURACY_H__ 

#include <cub/cub.cuh>

/**
 * 	@fn scoreacc_logistic_kernel
 * 	@brief accuracy score for logistic regression
 * */
__global__ void scoreacc_1d_logistic_kernel(const int, const float,
	const float*, const float*, float*); 

float scoreacc_1d_logistic(const int, const float, const float*, const float*,
	const int Mx=128);

#endif // __ACCURACY_H__
