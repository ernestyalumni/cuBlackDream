/**
 * @file   : linreg.cu
 * @brief  : Linear Regression with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  class CSVRow
 * 
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
 * nvcc -std=c++14 -lcublas ../src/FileIO/FileIO.cpp ../src/Axon/Axon.o ../src/smartptr/smartptr.cu ../src/Feedfwd/Feedfwd.cu linreg.cu -o linreg.exe
 * 
 * */
 
#include "../src/FileIO/FileIO.h"			// csv2fvec
#include "../src/Axon/Axon.h"				// Axon_sh
#include "../src/smartptr/smartptr.h"		// smartptr::RRModule
#include "../src/Feedfwd/Feedfwd.h"			// LinReg

#include <stdio.h> // printf

/** 
 *  @fn addb_explicit
 *  @brief I explicitly show how to add bias to the "output" layer
 *  @details Given (a_l)_i^{\  \  j} \in \text{Mat}_{\mathbb{R}}(m, s_l), 
 * 				we want to add a bias b, but along the "columns", b=b^j
 * 				assume (a_l) is COLUMN-major ordered.  
 *  			it is reasonable to assume m > s_l 
 * 				(i.e. number of rows, m, also representing the number of input examples, 
 * 				s_l = size dims. of "layer" l, a_l, or number of "nodes" of a_l
 * */

__global__ void addb_explicit(const int m, const int s_l, float* a_l, const float* b) {
	int k = threadIdx.x + blockDim.x * blockIdx.x ; // k is global thread index 

	// assume COLUMN-major ordering 
//	int i = k % m; 	// this the ith index in a matrix a_l(i,j)
	int j = k/m; 	// this is the jth index in matrix a_l(i,j)

	__shared__ float sh_bj[1]; // shared bj, jth component of b
	
	if ( j >= s_l) // check if j is access element outside of b\in \mathbb{R}^{s_l}
	{ return; } else { 
		sh_bj[0] = b[j]; 
	}
	int SIZEDIM_A_L = m*s_l;
	/* check if we've launched too many threads than needed to compute
	 * over all of s_l \in \text{Mat}_{\mathbb{R}}(m,s_l) - this could be the case 
	 * given arbitrary thread blocks launched in <<<>>>
	 * */
	if (k >= SIZEDIM_A_L)  
	{
		return ; 
	}
	__syncthreads();
	
	a_l[k] = a_l[k] + sh_bj[0];
	
}

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
	
	Axon Thetab(d,K);
	// Initialize fitting parameters with 0
	std::vector<float> h_Theta(d*K, 0.f);
	std::vector<float> h_b(K,0.f );
	std::vector<std::vector<float>> h_Thetab;
	h_Thetab.push_back(h_Theta);
	h_Thetab.push_back(h_b);

	Thetab.load_from_hvec(h_Theta,h_b);	// load the initial values of Theta, b
	Thetab.load_from_hXvec(X_ex1data1_colmaj,m); // load the X data
	Thetab.init_al(m);	// initialize the output layer


	/* ========== Compute cost functional J and residuals ========== */
	Thetab.rightMul();

	/* ========== Add bias ========== */
	
	/* ===== explicit implementation ===== */
	// this WORKS
/*	auto al = Thetab.getal();
	auto b = Thetab.getb();
	auto sizeDims = Thetab.getSizeDims();
	int SIZEDIM_A_L = m* sizeDims[1];
	int M_x = 128; // number of threads in a single thread block in x-direction
	addb_explicit<<<(SIZEDIM_A_L + M_x -1)/M_x, M_x>>>( m, sizeDims[1], al.get(), b.get());
*/
	int M_x = 128; // number of threads in a single thread block in x-direction
	Thetab.addb(M_x);

	// wrap up the y data
	smartptr::RRModule ptr_y( m*K);
	ptr_y.load_from_hvec( y_ex1data1_colmaj);

// this WORKS	
//	auto res = ptr_y.get(); // residual
	auto yhat = Thetab.getal();
	// custom deleter as a STRUCT for cublasHandle 
	struct del_cublasHandle_struct {
		void operator()(cublasHandle_t* ptr) { cublasDestroy(*ptr); }
	};
	
	std::unique_ptr<cublasHandle_t,del_cublasHandle_struct> handle_u(
		new cublasHandle_t);
	cublasCreate(handle_u.get());	
	
	// this WORKS
//	float a1 = -1.0f;
	const int SIZEDIM_Y = m*K;



	std::unique_ptr<float[], deleterRR_struct> res(new float[SIZEDIM_Y], deleterRR_struct());
	cudaMallocManaged((void **) &res, SIZEDIM_Y*sizeof(float));
	float a1 = 1.0f;
	float bet = -1.0f; 
	cublasSgeam(*handle_u.get(), 
		CUBLAS_OP_N, CUBLAS_OP_N, m, K, &a1, 
		yhat.get(), 
//		m, &bet, y_data.get(), m, 
		m, &bet, ptr_y.get().get(), m, 
		res.get(), m );


	// this WORKS
//	cublasSaxpy(*handle_u.get(), SIZEDIM_Y, &a1, yhat.get(), 1, res.get(), 1); 


	float costJ=0.f;
	cublasSnrm2(*handle_u.get(), SIZEDIM_Y, res.get(), 1, &costJ);
	costJ = 0.5f*costJ*costJ/ ((float) m);
	
	std::cout << " costJ : " << costJ << std::endl;
	
	auto a0 = Thetab.getalm1();
	
//	cublasSgemm(*handle_u.get(), CUBLAS_OP_T, CUBLAS_OP_N,d,K,m,&a1, a0.get(), 

	// WORKS but...
//	std::vector<Axon> axons;
	// DOES NOT WORK
//	axons.push_back(Thetab);
	// DOES NOT WORK
//	std::vector<Axon> axons = { Thetab };
	
	std::vector<int> FFsizeDims = { d,K }; 

	LinReg linreg( FFsizeDims );

	linreg.load_from_hThetaBs(h_Thetab);
	linreg.load_y_from_hvec(y_ex1data1_colmaj);

	// sanity check
	for (auto ele : y_ex1data1_colmaj) { std::cout << ele << " " ; } std::cout << std::endl;

	std::cout << " SIZE_X = h_Xvec.size() : " << X_ex1data1_colmaj.size() << std::endl;
//	for (auto ele : X_ex1data1_colmaj) { std::cout << ele << " "; }

	linreg.load_X_from_hvec(X_ex1data1_colmaj, m );

	linreg.feedfwd();

	/* this WORKS
	auto ycheck= std::move( linreg.gety() );
//	auto ycheck = linreg.gety();

	std::vector<float> hycheck(SIZEDIM_Y,0.f);
	cudaMemcpy(hycheck.data(), ycheck.get(), SIZEDIM_Y *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : hycheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	float result_linregcost = 0.f; 
	result_linregcost = linreg.compute_costJ_L2norm();
	
	std::cout << " result of linReg cost : " << result_linregcost << std::endl;


	// this single line of code WORKS
//	linreg.grad_desc_step(0.01f, 128);

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

	// this code WORKS
	cudaEventRecord(starttiming,0);
	linreg.grad_desc(1500,0.01f, 128);
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




}
