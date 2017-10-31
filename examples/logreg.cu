/**
 * @file   : logreg.cu
 * @brief  : Logistic Regression with CUDA C++14, CUBLAS, CUDA Unified Memory Management
 * @details :  class CSVRow
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171021  
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
 * nvcc -arch='sm_52' -std=c++14 -lcublas ../src/Feedfwd/Axon.o ../src/Feedfwd/activationf.o ../src/Feedfwd/Feedfwd.o ../src/FileIO/FileIO.cpp logreg.cu -o logreg.exe
 * 
 * */
 
#include <string> 	// std::string

#include "../src/FileIO/FileIO.h"			// csv2fvec, h_flatten_colmaj
#include "../src/Feedfwd/Feedfwd.h"

#include "../src/Axon/activationf.h"

#include <stdio.h> // printf

 
int main(int argc, char* argv[]) { 
	/** ==================== timing CUDA operations ==================== 
	 * @ref https://stackoverflow.com/questions/7876624/timing-cuda-operations
	 * */
	float timeinterval; 
	cudaEvent_t starttiming, stoptiming;
	cudaEventCreate(&starttiming);
	cudaEventCreate(&stoptiming);


	/* =============== ex2data1.txt =============== */ 
	std::string filename_ex2data1 = "../data/ex2data1.txt";
	auto fvec_ex2data1 = csv2fvec( filename_ex2data1) ;
	std::vector<std::vector<float>> X_ex1data2; // choose this type, than std::vector<float> so to generalize to multi-dim. case
	std::vector<std::vector<float>> y_ex1data2;
	for (auto row : fvec_ex2data1) { 
		std::vector<float> X_i = { row[0], row[1] }; // first 2 "columns" is input data X \in \mathbb{R}^m, d=2 features
		std::vector<float> y_i = { row[2] }; // second "column" is output data y \in \mathbb{R}^m, K=1 dim.
		X_ex1data2.push_back(X_i);	
		y_ex1data2.push_back(y_i);	
	}

	int d = X_ex1data2[0].size(); // number of features
	int K = y_ex1data2[0].size(); // dim. of output
	int m = X_ex1data2.size(); 	// m = number of training examples
	
	std::cout << std::endl << " For ex2data1.txt : " << std::endl; 
	std::cout << " d = " << d << ", K = " << K << ", m = " << m << std::endl;

	auto X_ex2data1_colmaj = h_flatten_colmaj(X_ex1data2); 
	auto y_ex2data1_colmaj = h_flatten_colmaj(y_ex1data2); 

	/* sanity check 
	for (auto ele : X_ex2data1_colmaj) { 
	std::cout << ele << " " ; }
	*/
	
	std::vector<int> FFsizeDims = { d,K }; 
	// pick sigmoid function
	std::vector<int> FFactfs = { 1 };

	LogisticReg logreg( FFsizeDims, FFactfs );

	// Initialize fitting parameters with 0
	std::vector<float> h_Theta(d*K, 0.f);
	std::vector<float> h_b(K,0.f );
	std::vector<std::vector<float>> h_Thetab;
	h_Thetab.push_back(h_Theta);
	h_Thetab.push_back(h_b);

	logreg.load_from_hThetaBs( h_Thetab);
	
	// sanity check
	// it WORKS
//	for (auto ele : y_ex2data1_colmaj) { std::cout << ele << " " ; } std::cout << std::endl;

	logreg.load_y_from_hvec(y_ex2data1_colmaj);

	// sanity check
	// it WORKS
/*	auto ycheck = logreg.gety();
	std::vector<float> hycheck(m*K,0.f);
	cudaMemcpy(hycheck.data(), ycheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : hycheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	logreg.load_X_from_hvec(X_ex2data1_colmaj, m);
	// sanity check
/*	auto alm1check = logreg.getalm1(1);
	std::vector<float> halm1check(m*d,0.f);
	cudaMemcpy(halm1check.data(), alm1check.get(), m*d *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : X_ex2data1_colmaj) { std::cout << ele << " " ; } std::cout << std::endl;
	for (auto ele : halm1check) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/
	// sanity check
/*	auto alcheck = logreg.getal(1);
	sigmoid_kernel<<<(m*K+128-1)/128,128>>>(m*K, alcheck.get()) ;
	std::vector<float> halcheck(m*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	logreg.feedfwd(256);
	// sanity check
	// it WORKS
/*	auto alcheck = logreg.getal(1);
	std::vector<float> halcheck(m*K,0.f);
	cudaMemcpy(halcheck.data(), alcheck.get(), m*K *sizeof(float),cudaMemcpyDeviceToHost);
	for (auto ele : halcheck) { 
		std::cout << ele << " " ; } std::cout << std::endl;
*/

	// sanity check
	float result_logregcost = 0.f; 
	result_logregcost = logreg.compute_costJ_xent(128);
	std::cout << " costJ for cross-entropy function, at initial theta (zeros) : " << result_logregcost << std::endl;
	std::cout << " Expected cost (approx): 0.693" << std::endl; 

	logreg.grad_desc_step(0.01f, 256);

	/* sanity check of 1 gradient descent  
	 * this (block of code) WORKS
	 * */
/*	auto Theta1 = std::move( logreg.getTheta(1) );
	auto b1 = std::move( logreg.getb(1) );
	std::vector<float> hTheta1(d*K,-1.f);
	std::vector<float> hb1(K,-1.f);
	cudaMemcpy(hTheta1.data(), Theta1.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb1.data(), b1.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta1 : " << hTheta1[0] << " " << hTheta1[1] << std::endl;
	std::cout << " hb1 : " << hb1[0] << " " << std::endl;
*/

	/* ========== Compute and display cost and gradient with non-zero theta ========== */
	std::vector<float> h_testTheta { 0.2f, 0.2f }; 
	std::vector<float> h_testb { -24.f };
	std::vector<std::vector<float>> h_testThetab;
	h_testThetab.push_back(h_testTheta);
	h_testThetab.push_back(h_testb);

	logreg.load_from_hThetaBs( h_testThetab);
	logreg.feedfwd(256);
	float result_testlogregcost = logreg.compute_costJ_xent(256);

	std::cout << std::endl << " Cost at test theta: " << result_testlogregcost << std::endl;
	std::cout << " Expected cost (approx): 0.218 " << std::endl; 

	logreg.grad_desc_step(0.01f, 256);

	/* sanity check of 1 gradient descent  
	 * this (block of code) WORKS
	 * */
/*	auto Theta1 = std::move( logreg.getTheta(1) );
	auto b1 = std::move( logreg.getb(1) );
	std::vector<float> hTheta1(d*K,-1.f);
	std::vector<float> hb1(K,-1.f);
	cudaMemcpy(hTheta1.data(), Theta1.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb1.data(), b1.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta1 : " << hTheta1[0] << " " << hTheta1[1] << std::endl;
	std::cout << " hb1 : " << hb1[0] << " " << std::endl;
*/
	// fprintf('Expected gradients (approx):\n 0.043\n 2.566\n 2.647\n');
	logreg.load_from_hThetaBs( h_Thetab);
	logreg.feedfwd(256);


	cudaEventRecord(starttiming,0);
	logreg.grad_desc(1500000,0.0010f, 256);
	cudaEventRecord(stoptiming,0);
	cudaEventSynchronize(stoptiming);
	cudaEventElapsedTime(&timeinterval, starttiming,stoptiming);
	printf("Time to grad_desc 1500 iterations : %3.1f ms \n ", timeinterval);

	result_testlogregcost = logreg.compute_costJ_xent(256);


	std::cout << " Cost at theta found by grad desc: " << result_testlogregcost << std::endl; 
	std::cout << " Expected cost (approx): 0.203 " << std::endl; 


	/* sanity check of gradient descent  
	 * this (block of code) WORKS
	 * */
	auto Theta1 = std::move( logreg.getTheta(1) );
	auto b1 = std::move( logreg.getb(1) );
	std::vector<float> hTheta1(d*K,0.f);
	std::vector<float> hb1(K,0.f);
	cudaMemcpy(hTheta1.data(), Theta1.get(), d*K*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(hb1.data(), b1.get(), K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout << " hTheta1 : " << hTheta1[0] << " " << hTheta1[1] << std::endl;
	std::cout << " hb1 : " << hb1[0] << " " << std::endl;

	// fprintf('Expected theta (approx):\n');
	// fprintf(' -25.161\n 0.206\n 0.201\n');



}

