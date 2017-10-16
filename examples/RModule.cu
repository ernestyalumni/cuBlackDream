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
 * nvcc -std=c++14 -lcublas ../src/Axon/Axon.cu RModule.cu -o Rmodule.exe
 * 
 * */
#include "../src/Axon/Axon.h"				// Axon_sh
//#include "../src/smartCUBLAS/smartCUBLAS.h"	 // Prod_sh

//#include "cublas_v2.h" 

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
	for (i=0;i<n;i++) { b_vec[i] = ((float) i + 1); }

	Axon_sh Rmodule(k,n);
	Rmodule.load_from_hvec(b,b_vec);
	Rmodule.load_from_hXvec(a,m);
	Rmodule.init_al(m);

	// unique_ptr version
	Axon Rmodule_u(k,n);
	Rmodule_u.load_from_hvec(b,b_vec);
	Rmodule_u.load_from_hXvec(a,m);
	Rmodule_u.init_al(m);


	std::vector<int> sizeDims = Rmodule_u.getSizeDims();
	
	const int s_lm1 = sizeDims[0];
	const int s_l= sizeDims[1];
	const int mm= sizeDims[2];
	std::cout << " s_lm1 : " << s_lm1 << " s_l : " << s_l << " m : " << mm << std::endl;

/*
	auto a_u = std::move( Rmodule_u.getalm1() );  
	auto b_u = std::move( Rmodule_u.getTheta() );
	auto c_u = std::move( Rmodule_u.getal() );

	struct del_cublasHandle_struct {
		void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
	};

	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());

	float a1=1.0f;
	float bet = 0.f;
	cublasSgemm(*handle_u.get(),CUBLAS_OP_N,CUBLAS_OP_N,mm,s_l,s_lm1,&a1,a_u.get(),mm,b_u.get(),s_lm1,&bet,c_u.get(),mm);

	std::vector<float> utempal(m*n);
	cudaMemcpy(utempal.data(), c_u.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);


	// print c row by row
	std::cout << "c_u: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << utempal[i +j*m] << " ";
		}
		std::cout << std::endl;
	}
*/


	
//	cudaDeviceSynchronize();
	Rmodule_u.rightMul();
	auto al_sh = std::move( Rmodule_u.getal() );
	std::vector<float> tempal_sh(m*n);
	cudaMemcpy(tempal_sh.data(), al_sh.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);

	// print c row by row
	std::cout << "c_general: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << tempal_sh[i +j*m] << " ";
		}
		std::cout << std::endl;
	}


//	rightMul(Rmodule_u);

	// sanity check
	std::vector<float> tempTheta(k*n); 
	std::vector<float> tempb(n);
//	std::vector<float> tempTheta; 
//	std::vector<float> tempb;

//	auto Thetaptr = std::move( Rmodule.getTheta() )
//	cudaMemcpy(tempTheta.data(), Thetaptr.get(), sizeof(float) * k*n, cudaMemcpyDeviceToHost);

	// sanity check
	Rmodule.load_from_d(tempTheta, tempb);
	for (auto ele : tempTheta) { std::cout << ele << " "; } std::cout << std::endl << std::endl;
	for (auto ele : tempb) { std::cout << ele << " "; } std::cout << std::endl; 


	Rmodule.rightMul();

	auto a_sh = std::move( Rmodule.getalm1() );  
	auto b_sh = std::move( Rmodule.getTheta() );
	auto c_sh = std::move( Rmodule.getal() );
	
/*	Prod_sh( m,n,k, 1.0f, a_sh, b_sh, 
		0.f, c_sh ); */ // needs smartCUBLAS
		

	std::vector<float> tempal(m*n);
	cudaMemcpy(tempal.data(), c_sh.get(), sizeof(float)*m*n, cudaMemcpyDeviceToHost);

	// print c row by row
	std::cout << "c_sh: " << std::endl;
	for (i=0;i<m;i++){
		for (j=0;j<n;j++) {
			std::cout << tempal[i +j*m] << " ";
		}
		std::cout << std::endl;
	}


	// I loaded the above 4 commands into member function rightMul of Axon_sh
//	Rmodule.rightMul();
	
}
