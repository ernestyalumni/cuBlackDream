/**
 * @file   : accuracy.cu
 * @brief  : scoring accutacy functions content/source file in CUDA C++14, 
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
 * nvcc -std=c++11 -arch='sm_61' -I./cub-1.6.4/ -dc accuracy.cu -o accuracy.o
 * 
 * */
#include "accuracy.h"

__global__ void scoreacc_1d_logistic_kernel(const int SIZE, const float threshold, 
	const float* y, const float* yhat, float* score) 
{
	int tid = threadIdx.x + blockDim.x * blockIdx.x;  
	
	if (tid >= SIZE) { return; } 

	for (int kx=tid; kx < SIZE; kx += gridDim.x*blockDim.x) 
	{			
		y_val = y[tid];
		yhat_val = yhat[tid];
		if (yhat_val > threshold) {
			yhat_val = 1.f; 
		} else {
			yhat_val = 0.f; 
		}
		score_val = ((float) (yhat_val == y_val)); 
		score[tid] = score_val;
	}		
}

float scoreacc_1d_logistic(const int m, const float threshold, const float *y, const float*yhat, 
	const int Mx) 
{
	auto deleter=[&](float* ptr){ cudaFree(ptr); };
	std::unique_ptr<float[], decltype(deleter)> d_scores(new float[m], deleter);
	cudaMallocManaged((void **) &d_scores,m*sizeof(float));

	std::unique_ptr<float, decltype(deleter)> d_sum(new float, deleter);
	cudaMallocManaged((void **) &d_sum,1*sizeof(float));


	int Nx = (m + Mx - 1)/Mx;
	scoreacc_1d_logistic_kernel<<<Nx,Mx>>>(m,threshold,y,yhat,d_scores.get() );
	


	// Request and allocate temporary storage
	size_t temp_storage_bytes =0;
	float* d_temp_storage=nullptr;

	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_scores.get(), &d_sum, m );

	CubDebugExit( cudaMallocManaged(&d_temp_storage, temp_storage_bytes) );

	// Run
	cub::DeviceReduce::Sum( d_temp_storage, temp_storage_bytes, d_scores.get(), &d_sum, m) ;

	float result = d_sum / ((float) m );

	return result;
}



