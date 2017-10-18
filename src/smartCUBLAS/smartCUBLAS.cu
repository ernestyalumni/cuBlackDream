/**
 * @file   : smartCUBLAS.cu
 * @brief  : Smart pointers content/source file in CUDA C++14, 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171015  
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
 * nvcc -std=c++14 -dc Axon.cu -o Axon.o
 * 
 * */
#include "smartCUBLAS.h"

/* =============== Matrix Multiplication or contraction as a function  =============== */

/**
 *  @fn	Prod_sh
 * 	@brief Matrix Multiplication product with std::shared_ptrs
 *  @details Level-3 SGEMM - matrix-matrix multiplication
 * 				C = \alpha op(A) op(B) + \beta C
 * where A,B are matrices in column-major format, \alpha, beta are scalars
 * 			
 * */
void Prod_sh(const int m, const int n, const int k, 
		const float a1, std::shared_ptr<float>& A, std::shared_ptr<float>& B, 
		const float bet, std::shared_ptr<float>& C) {
	
	auto del_cublasHandle=[&](cublasHandle_t* ptr) { cublasDestroy(*ptr); };

	// moved the cublasHandle_t environment into the product itself
	std::shared_ptr<cublasHandle_t> handle_sh(
		new cublasHandle_t, del_cublasHandle);
	cublasCreate(handle_sh.get());
			
	cublasSgemm(*handle_sh.get(),CUBLAS_OP_N,CUBLAS_OP_N,m,n,k,&a1,A.get(),m,B.get(),k,&bet,C.get(),m);
					
}
