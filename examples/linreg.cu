/**
 * @file   : linreg.cu
 * @brief  : Linear Regression with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  class CSVRow
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171014  
 * @ref    : Coursera, Andrew Ng, Intro. to Machine Learning, ex1, Exercises 1 of Week 2
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
 * First do, in cuBlackDream/src/Feedfwd/ folder, 
 * nvcc -std=c++14 -arch='sm_52' -lcublas -dc ../Axon/Axon.cu ../Axon/activationf.cu Feedfwd.cu
 * 
 * Then, in the folder of this, linreg.cu, do the following:
 * nvcc -arch='sm_52' -std=c++14 -lcublas ../src/Feedfwd/Axon.o ../src/Feedfwd/activationf.o ../src/Feedfwd/Feedfwd.o 
 * 	../src/FileIO/FileIO.cpp linreg.cu -o linreg.exe
 * */
 
#include "../src/FileIO/FileIO.h"			// csv2fvec
#include "../src/Feedfwd/Feedfwd.h"			// LinReg

#include <stdio.h> // printf


int main(int argc, char* argv[]) { 
	/** ==================== timing CUDA operations ==================== 
	 * @ref https://stackoverflow.com/questions/7876624/timing-cuda-operations
	 * */
	float timeinterval; 
	cudaEvent_t starttiming, stoptiming;
	cudaEventCreate(&starttiming);
	cudaEventCreate(&stoptiming);
	


	/* =============== ex1data1.txt =============== */ 
	std::string filename_ex1data1 = "../data/ex1data1.txt";
	
	auto Xydata_ex1data1 = csv2fvec(filename_ex1data1); 
	
	std::vector<std::vector<float>> X_ex1data1; // choose this type, than std::vector<float> so to generalize to multi-dim. case
	std::vector<std::vector<float>> y_ex1data1;
	for (auto row : Xydata_ex1data1) { 
		std::vector<float> X_i = { row[0] }; // first "column" is input data X \in \mathbb{R}^m, d=1 features
		std::vector<float> y_i = { row[1] }; // second "column" is output data y \in \mathbb{R}^m, K=1 dim.

		X_ex1data1.push_back(X_i);	
		y_ex1data1.push_back(y_i);	
	}
	
	// notice that both X_ex1data1, y_ex1data1 are row-major ordered.  We want column-major ordering for CUBLAS
	
		
	int d = X_ex1data1[0].size(); // number of features
	int K = y_ex1data1[0].size(); // dim. of output
	int m = X_ex1data1.size(); 	// m = number of training examples

	std::cout << " d : " << d << std::endl;
	std::cout << " K : " << K << std::endl;
	std::cout << " m : " << m << std::endl;


	// preprocess X_ex1data1
	// flatten X_ex1data1 into column-major ordering 
//	auto X_ex1data1_colmaj = h_flatten_colmaj( X_ex1data1 );
	std::vector<float> X_ex1data1out;
	for (int j=0; j<d; j++){ 
		for (int i=0; i<m; i++) {
			X_ex1data1out.push_back( (X_ex1data1[i])[j] ); 
		}
	}
	std::cout << std::endl << " X_ex1data1out.size() : " << X_ex1data1out.size() << std::endl;


	std::vector<float> X_ex1data1_colmaj = h_flatten_colmaj( X_ex1data1 );
	auto y_ex1data1_colmaj = h_flatten_colmaj( y_ex1data1 ); 
	

	// Initialize fitting parameters with 0
	std::vector<float> h_Theta(d*K, 0.f);
	std::vector<float> h_b(K,0.f );
	std::vector<std::vector<float>> h_Thetab;
	h_Thetab.push_back(h_Theta);
	h_Thetab.push_back(h_b);

	
	std::vector<int> FFsizeDims = { d,K }; 

	LinReg linreg( FFsizeDims );


	linreg.load_from_hThetaBs(h_Thetab);
	linreg.load_y_from_hvec(y_ex1data1_colmaj);


	std::cout << " SIZE_X = h_Xvec.size() : " << X_ex1data1_colmaj.size() << std::endl;


	linreg.load_X_from_hvec(X_ex1data1_colmaj, m );

	linreg.feedfwd(128);

	float result_linregcost = 0.f; 
	result_linregcost = linreg.compute_costJ_L2norm();
	
	std::cout << " result of linReg cost : " << result_linregcost << std::endl;


	// this single line of code WORKS
	linreg.grad_desc_step(0.01f, 128);

	/* sanity check of 1 gradient descent  
	 * this (block of code) WORKS
	 * */
/*	auto Theta1 = std::move( linreg.getTheta(1) );
	auto b1 = std::move( linreg.getb(1) );
	std::vector<float> hTheta1(d*K,0.f);
	std::vector<float> hb1(K,0.f);
	cudaMemcpy(hTheta1.data(), Theta1.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb1.data(), b1.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta1 : " << hTheta1[0] << " " << std::endl;
	std::cout << " hb1 : " << hb1[0] << " " << std::endl;
*/
	
	/* ===== check if dJ/db calculation works ===== */
	// in this scope, make res to store results from taking the difference
	/* it WORKS /*
	struct del_cublasHandle_struct {
		void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
	};
	const int SIZE_Y= K * m; 
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	
	std::unique_ptr<float[], deleterRR_struct> res(new float[SIZE_Y], deleterRR_struct());
	cudaMallocManaged((void **) &res, SIZE_Y*sizeof(float));
	float a1 = 1.0f;
	float bet = -1.0f; 
	auto yhat = linreg.getal(1); // L+1 - 2 = L-1 which is the last axon, when counting from 0, 0,1,...L-1
	auto y = std::move(linreg.gety());
	cublasSgeam(*handle_u.get(), 
		CUBLAS_OP_N, CUBLAS_OP_N, m, K, &a1, 
		yhat.get(), m, &bet, y.get(), m, 
		res.get(), m );
	std::vector<float> hres( SIZE_Y, 2.0f);
	cudaMemcpy(hres.data(), res.get(), m*K*sizeof(float), cudaMemcpyDeviceToHost);
	for (auto ele : hres) { std::cout << ele << " "; }  std::cout << std::endl << std::endl;
	a1 = 1.0f/ ((float) m);
	bet = 0.f;
	const int SIZE_dB = K;
	std::unique_ptr<float[], deleterRR_struct> dB(new float[SIZE_dB], deleterRR_struct());
	cudaMallocManaged((void **) &dB, SIZE_dB*sizeof(float));
	// create 1s array, array of 1s
	const int SIZE_ONES = m;
	std::unique_ptr<float[], deleterRR_struct> ones(new float[SIZE_ONES], deleterRR_struct());
	cudaMallocManaged((void **) &ones, SIZE_ONES*sizeof(float));
	int Mx=128;
	int Nx = (SIZE_ONES + Mx - 1)/Mx; 
//	if ( MAX_SIZE_1DARR < SIZE_ONES ) {
//		Nx = (MAX_SIZE_1DARR + Mx - 1) / Mx ; } 
	setconstval_kernel<<<Nx,Mx>>>(m,1.0f, ones.get() );
	// this is a clever way to do summation
	cublasSgemm(*handle_u.get(), CUBLAS_OP_N, CUBLAS_OP_N, 1,K,m, 
		&a1, ones.get(), 1, res.get(), m, 
		&bet, dB.get(), 1); 
	std::vector<float> hones( m, 2.5f);
	std::vector<float> hdB( K, 2.5f);
	cudaMemcpy(hdB.data(), dB.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hones.data(), ones.get(), m*sizeof(float), cudaMemcpyDeviceToHost);
	for (auto ele : hdB) { std::cout << ele << " "; } std::cout << std::endl << std::endl;
	for (auto ele : hones) { std::cout << ele << " "; } std::cout << std::endl << std::endl;
*/
	

	// this code WORKS
	cudaEventRecord(starttiming,0);
	linreg.grad_desc(1500,0.01f, 256);
	cudaEventRecord(stoptiming,0);
	cudaEventSynchronize(stoptiming);
	cudaEventElapsedTime(&timeinterval, starttiming,stoptiming);
	printf("Time to grad_desc 1500 iterations : %3.1f ms \n ", timeinterval);

	/* sanity check of gradient descent  
	 * this (block of code) WORKS
	 * */
	auto Theta1 = std::move( linreg.getTheta(1) );
	auto b1 = std::move( linreg.getb(1) );
	std::vector<float> hTheta1(d*K,0.f);
	std::vector<float> hb1(K,0.f);
	cudaMemcpy(hTheta1.data(), Theta1.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb1.data(), b1.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta1 : " << hTheta1[0] << " " << std::endl;
	std::cout << " hb1 : " << hb1[0] << " " << std::endl;


	/* =============== multi-dim. linear regression case =============== */

	/* =============== ex1data2.txt =============== */ 
	std::string filename_ex1data2 = "../data/ex1data2.txt";
	
	auto Xydata_ex1data2 = csv2fvec(filename_ex1data2); 
	
	std::vector<std::vector<float>> X_ex1data2; // choose this type, than std::vector<float> so to generalize to multi-dim. case
	std::vector<std::vector<float>> y_ex1data2;
	for (auto row : Xydata_ex1data2) { 
		std::vector<float> X_i = { row[0], row[1] }; // first "column" is input data X \in \mathbb{R}^m, d=2 features
		std::vector<float> y_i = { row[2] }; // second "column" is output data y \in \mathbb{R}^m, K=1 dim.

		X_ex1data2.push_back(X_i);	
		y_ex1data2.push_back(y_i);	
	}
	
	// notice that both X_ex1data2, y_ex1data2 are row-major ordered.  We want column-major ordering for CUBLAS
	
	d = X_ex1data2[0].size(); // number of features
	K = y_ex1data2[0].size(); // dim. of output
	m = X_ex1data2.size(); 	// m = number of training examples

	std::cout << std::endl << " multi-dim. linear reg case : " << std::endl;
	std::cout << " d : " << d << std::endl;
	std::cout << " K : " << K << std::endl;
	std::cout << " m : " << m << std::endl;

	// preprocess X_ex1data2
	// flatten X_ex1data2 into column-major ordering 
	std::vector<float> X_ex1data2out;
	for (int j=0; j<d; j++){ 
		for (int i=0; i<m; i++) {
			X_ex1data2out.push_back( (X_ex1data2[i])[j] ); 
		}
	}
	std::cout << std::endl << " X_ex1data2out.size() : " << X_ex1data2out.size() << std::endl;

	std::vector<float> X_ex1data2_colmaj = h_flatten_colmaj( X_ex1data2 );
	auto y_ex1data2_colmaj = h_flatten_colmaj( y_ex1data2 ); 

	/* ========== Feature Normalize in Python, then numpy.array.tofile -> std::vector<float> ========== */
	std::string filename_Xex1data2_npy = "../data/Xex1data2.npy";
	auto X_ex1data2_bin = npy2fvec(filename_Xex1data2_npy, m, d);


	// Initialize fitting parameters with 0
	std::vector<float> h_Theta_multi(d*K, 0.f);
	std::vector<float> h_b_multi(K,0.f );
	std::vector<std::vector<float>> h_Thetab_multi;
	h_Thetab_multi.push_back(h_Theta_multi);
	h_Thetab_multi.push_back(h_b_multi);


	std::vector<int> FFsizeDims_multi = { d,K }; 

	LinReg multilinreg( FFsizeDims_multi );

	multilinreg.load_from_hThetaBs(h_Thetab_multi);
	multilinreg.load_y_from_hvec(y_ex1data2_colmaj);
	multilinreg.load_X_from_hvec(X_ex1data2_bin, m );

	cudaEventRecord(starttiming,0);
	multilinreg.grad_desc(400,0.01f, 128);
	cudaEventRecord(stoptiming,0);
	cudaEventSynchronize(stoptiming);
	cudaEventElapsedTime(&timeinterval, starttiming,stoptiming);
	printf("Time to multi-dim. grad_desc 400 iterations : %3.1f ms \n ", timeinterval);

	multilinreg.grad_desc(10000,0.001f, 128);


	/* sanity check of gradient descent  
	 * this (block of code) WORKS
	 * */
	auto Theta_multi = std::move( multilinreg.getTheta(1) );
	auto b_multi = std::move( multilinreg.getb(1) );
	std::vector<float> hTheta_multi(d*K,0.f);
	std::vector<float> hb_multi(K,0.f);
	cudaMemcpy(hTheta_multi.data(), Theta_multi.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb_multi.data(), b_multi.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta (multi) : " << hTheta_multi[0] << " " << hTheta_multi[1] << std::endl;
	std::cout << " hb (multi) : " << hb_multi[0] << " " << std::endl;

	

}
