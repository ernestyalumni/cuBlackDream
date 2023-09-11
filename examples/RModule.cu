/**
 * @file   : RModule.cu
 * @brief  : RModule Abstract Algebra with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  R-Module Abstract Algebra.  
 * 
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171015  
 * @ref    : 
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
 * nvcc -std=c++14 -arch='sm_52' -lcublas ../src/Axon/Axon.cu ../src/Axon/activationf.cu RModule.cu -o RModule.exe
 * */
#include "../src/Axon/Axon.h"				

#include <iostream>

int main(int argc, char* argv[]) { 
	constexpr const int m=6; 								// a - mxk matrix
	constexpr const int n=4;								// b - kxn matrix
	constexpr const int k=5;								// c - mxn matrix	

	// Notice that in CUBLAS, matrices follow COLUMN-major ordering

	// "boilerplate" values on host
		// define an mxk matrix a column by column
	std::vector<float> a(m*k);
	std::vector<float> b(k*n);
	std::vector<float> c(m*n);

	int i,j;								// i-row index,j-column index

	int ind=11;								// a:
	for (j=0;j<k;j++) { 					// 11,17,23,29,35 
		for (i=0;i<m;i++) { 				// 12,18,24,30,36
			a[i +j*m]=(float) ind++;			// 13,19,25,31,37
		}									// 14,20,26,32,38
	}										// 15,21,27,33,39
											// 16,22,28,34,40
	
	// print a row by row
	std::cout << " a: " << std::endl;
	for (i=0;i<m;i++) {
		for (j=0;j<k;j++) {
			std::cout << a[i+j*m] << " " ; }
		std::cout << std::endl; 
	}
	// define a kxn matrix b column by column
	ind=11;									// b:
	for (j=0;j<n;j++) {						// 11,16,21,26
		for (i=0;i<k;i++) {					// 12,17,22,27
			b[i+j*k]=(float) ind++;			// 13,18,23,28 
		}									// 14,19,24,29
	}										// 15,20,25,30
	// print b row by row
	std::cout << " b: " << std::endl;
	for (i=0;i<k;i++) {
		for (j=0;j<n;j++) {
			std::cout << b[i+j*k] << " " ; } 
		std::cout << std::endl;
	}
	
	// define an mxn matrix c column by column
	ind=11;									// c:
	for (j=0;j<n;j++) {						// 11,17,23,29
		for (i=0;i<m;i++) {					// 12,18,24,30
			c[i+j*m]=(float)ind++;			// 13,19,25,31
		}									// 14,20,26,32
	}										// 15,21,27,33
											// 16,22,28,34
	// print c row by row
	std::cout << "c: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << c[i +j*m] << " ";
		}
		std::cout << std::endl;
	}

	// define a n-vector, b_vec \in \mathbb{R}^n
	std::vector<float> b_vec(n);
	for (i=0;i<n;i++) { b_vec[i] = ((float) i + 1)*11.f; }

	Axon Rmodule(k,n);
	
	Rmodule.load_from_hvec(b,b_vec);
	Rmodule.load_from_hXvec(a,m);
	Rmodule.init_al(m);

	std::vector<int> sizeDims = Rmodule.getSizeDims();
	
	const int s_lm1 = sizeDims[0];
	const int s_l= sizeDims[1];
	const int mm= sizeDims[2];
	std::cout << " s_lm1 : " << s_lm1 << " s_l : " << s_l << " m : " << mm << std::endl;


	Rmodule.rightMul();

	// this WORKS
	auto al_sh = Rmodule.getal();
	std::vector<float> tempal_sh(m*n);
	cudaMemcpy(tempal_sh.data(), al_sh.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);

	// print c row by row
	std::cout << "c, after Matrix Multiplication : " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << tempal_sh[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
	
	al_sh.reset();

	Rmodule.addb(128);

	auto al_bias = Rmodule.getal();
	std::vector<float> tempal_bias(m*n);
	cudaMemcpy(tempal_bias.data(), al_bias.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	// print c, after adding bias, row by row
	std::cout << "c, after adding bias : " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << tempal_bias[i +j*m] << " ";
		}
		std::cout << std::endl;
	}

	/* =============== troubleshooting adding bias b =============== */

	// sanity check on bias b
	auto gettingb = Rmodule.getb();
	std::vector<float> tempbias(n);
	cudaMemcpy(tempbias.data(), gettingb.get(), sizeof(float)*n, cudaMemcpyDeviceToHost);
	// print b, after adding bias, row by row
	std::cout << "bias b, after adding bias : " << std::endl;
	for (j=0;j<n;j++) {
		std::cout << tempbias[j] << " ";
	}
	std::cout << std::endl;

	/* ========================= Axon_act ========================= */

	std::cout << std::endl << " Doing it for Axon_act class : " << std::endl << std::endl;

	Axon_act Rmodule_act(k,n,0); // 0 for identity
	
	Rmodule_act.load_from_hvec(b,b_vec);
	Rmodule_act.load_from_hXvec(a,m);
	Rmodule_act.init_zlal(m);
	Rmodule_act.rightMul();

	// this WORKS
/*	auto zl_ptr = Rmodule_act.getzl();
	std::vector<float> tempzl_act(m*n, 1.f);
	cudaMemcpy(tempzl_act.data(), zl_ptr.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	std::cout << "z^l, after matrix multiplication  : " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << tempzl_act[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
*/

	Rmodule_act.addb(128);
	Rmodule_act.actf(128);
	Rmodule_act.do_Dpsi(128);

	auto al_ptr = Rmodule_act.getal();
	std::vector<float> al_vec(m*n, 1.1f);
	cudaMemcpy(al_vec.data(), al_ptr.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << al_vec[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
	
	auto Dpsi_ptr = Rmodule_act.getDpsil();
	cudaMemcpy(al_vec.data(), Dpsi_ptr.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << al_vec[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
}