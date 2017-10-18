/**
 * @file   : smartCUBLAS.h
 * @brief  : header file for Smart pointers (shared and unique ptrs) with CUBLAS, in C++14, 
 * @details : smart pointers with CUBLAS; 
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
 * nvcc -std=c++14 -lcublas ./smartptr/smartptr.cu smartCUBLAS_playground.cu -o smartCUBLAS_playground.exe
 * 
 * */
#include "cublas_v2.h" 

#include <memory> // std::shared_ptr


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
		const float bet, std::shared_ptr<float>& C);
		
/* =============== Matrix Addition or R-module addition =============== */
/**
 * 	@fn add_u_sh
 * 	@brief Add a unique ptr and shared ptr
 * */
